"""Robustness experiment: evaluate trained models on corrupted val images.

No training - loads existing best.pt checkpoints and measures how mIoU
degrades as test-time image corruptions intensify. This is the project's
third research axis: do CNN and Transformer differ in robustness to
distribution shift?

Corruptions (from the imagecorruptions package) at severity 1..5:
    gaussian_noise, motion_blur, fog, brightness
Fog and brightness are especially relevant to driving scenes.

The corruption is applied to the raw uint8 image BEFORE normalization, so
it simulates a genuinely degraded input. Labels are unchanged.

Usage (after both models are trained):
    python src/robustness.py --runs unet_100 segformer_100 --split val

Produces:
    outputs/analysis/robustness_<split>.csv       model x corruption x severity
    outputs/analysis/robustness_<split>.png        degradation curves
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- NumPy 2.0 compatibility shim --------------------------------------
# The imagecorruptions package (used below for fog/blur/noise) predates
# NumPy 2.0 and still references aliases that 2.0 removed (np.float_,
# np.int_, etc.). Re-point them to their canonical types so the legacy
# library runs unchanged on a modern NumPy. Harmless on older NumPy too.
for _alias, _target in (("float_", "float64"), ("int_", "int64"),
                        ("bool_", "bool_"), ("object_", "object_"),
                        ("complex_", "complex128")):
    if not hasattr(np, _alias) and hasattr(np, _target):
        setattr(np, _alias, getattr(np, _target))
# -----------------------------------------------------------------------

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import (
    CamVidDataset, IMAGENET_MEAN, IMAGENET_STD, VOID_INDEX,
)
from metrics import SegMetrics
from models import build_model

CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "brightness"]
SEVERITIES = [1, 2, 3, 4, 5]


class CorruptedCamVid(Dataset):
    """Val dataset that applies one corruption at one severity, then the
    normal padding + ImageNet normalization. Corruption is applied to the
    uint8 RGB image; the mask is only padded."""

    def __init__(self, root, corruption, severity, height=384, width=480):
        from imagecorruptions import corrupt
        self._corrupt = corrupt
        self.corruption = corruption
        self.severity = severity
        self.base = CamVidDataset(root, "val", transforms=None)
        self.pad = A.PadIfNeeded(height, width, border_mode=cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0), mask_value=VOID_INDEX)
        self.norm = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        name = self.base.names[i]
        image = cv2.cvtColor(cv2.imread(str(self.base.img_dir / name)),
                             cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.base.ann_dir / name), cv2.IMREAD_GRAYSCALE)
        # corrupt expects uint8 HxWx3; returns uint8
        if self.corruption is not None:
            image = self._corrupt(image, corruption_name=self.corruption,
                                  severity=self.severity).astype(np.uint8)
        padded = self.pad(image=image, mask=mask)
        out = self.norm(image=padded["image"])
        return out["image"], torch.from_numpy(padded["mask"]).long()


def load_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg["model"]


@torch.no_grad()
def eval_loader(model, loader, device):
    m = SegMetrics(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
        m.update(logits, y)
    return m.compute()["miou"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="run names to compare, e.g. unet_100 segformer_100")
    p.add_argument("--split", default="val")
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for r in args.runs:
        model, mtype = load_model(Path(args.runs_dir) / r, device)
        models[r] = (model, mtype)
        print(f"[robust] loaded {r} ({mtype})")

    # results[run][corruption] = [miou @ sev1..5]; also clean baseline.
    results = {r: {"clean": None} for r in args.runs}

    # clean baseline (severity 0)
    clean_ds = CamVidDataset(args.root, "val",
                             transforms=None)  # placeholder; use corrupted=None
    clean = CorruptedCamVid(args.root, None, 0)
    clean_loader = DataLoader(clean, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    for r, (model, _) in models.items():
        results[r]["clean"] = eval_loader(model, clean_loader, device)
        print(f"[robust] {r} clean mIoU = {results[r]['clean']:.4f}")

    for corruption in CORRUPTIONS:
        for r in args.runs:
            results[r][corruption] = []
        for sev in SEVERITIES:
            ds = CorruptedCamVid(args.root, corruption, sev)
            loader = DataLoader(ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
            for r, (model, _) in models.items():
                miou = eval_loader(model, loader, device)
                results[r][corruption].append(miou)
                print(f"[robust] {r}  {corruption}  sev{sev}  mIoU {miou:.4f}")

    # ---- CSV -----------------------------------------------------------
    csv_path = out_dir / f"robustness_{args.split}.csv"
    with open(csv_path, "w") as f:
        f.write("run,corruption,severity,miou\n")
        for r in args.runs:
            f.write(f"{r},clean,0,{results[r]['clean']:.4f}\n")
            for corruption in CORRUPTIONS:
                for sev, miou in zip(SEVERITIES, results[r][corruption]):
                    f.write(f"{r},{corruption},{sev},{miou:.4f}\n")
    print(f"[robust] wrote {csv_path}")

    # ---- plot: one subplot per corruption ------------------------------
    fig, axes = plt.subplots(1, len(CORRUPTIONS),
                             figsize=(5 * len(CORRUPTIONS), 4.5), sharey=True)
    for ax, corruption in zip(axes, CORRUPTIONS):
        for r in args.runs:
            ys = [results[r]["clean"]] + results[r][corruption]
            ax.plot([0] + SEVERITIES, ys, marker="o", label=r)
        ax.set_title(corruption)
        ax.set_xlabel("severity")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mIoU")
    axes[0].legend()
    fig.suptitle("Robustness to corruptions: CNN vs Transformer")
    fig.tight_layout()
    plot_path = out_dir / f"robustness_{args.split}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"[robust] wrote {plot_path}")

    # ---- relative degradation summary ----------------------------------
    print("\n  Mean relative mIoU retained at severity 5 "
          "(higher = more robust):")
    for r in args.runs:
        retained = np.mean([
            results[r][c][-1] / results[r]["clean"] for c in CORRUPTIONS
        ])
        print(f"    {r:<20} {100 * retained:5.1f}%  of clean mIoU")


if __name__ == "__main__":
    main()
