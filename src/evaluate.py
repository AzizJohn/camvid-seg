"""Evaluate a trained checkpoint on the val or test split.

Loads best.pt for a run, runs inference, prints the per-class IoU table and
mean IoU / pixel accuracy, saves a JSON of the numbers, and renders a grid
of qualitative predictions (image | ground truth | prediction).

IMPORTANT: do NOT evaluate on the test split before Day 8. Until then use
--split val. The test set is touched exactly once, at the end, per model.

Examples
--------
    python src/evaluate.py --run unet_100 --split val
    python src/evaluate.py --run segformer_100 --split val --num-vis 8

Reads:  outputs/runs/<run>/best.pt  (and config.json for the model type)
Writes: outputs/runs/<run>/eval_<split>.json
        outputs/runs/<run>/pred_<split>.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    CamVidDataset, get_transforms, decode_segmap,
    CLASS_NAMES, PALETTE, VOID_INDEX, NUM_CLASSES,
)
from metrics import SegMetrics, format_per_class
from models import build_model


def load_run_model(run_dir: Path, device: torch.device):
    """Rebuild the right architecture from config.json and load best.pt."""
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    model_name = cfg["model"]
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] {run_dir.name}: {model_name}, "
          f"best epoch {ckpt.get('epoch','?')}, "
          f"train-time val mIoU {ckpt.get('miou', float('nan')):.4f}")
    return model, model_name, cfg


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    metrics = SegMetrics(device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        metrics.update(logits, targets)
    return metrics.compute()


@torch.no_grad()
def make_vis(model, ds, device, n, out_path, use_amp=True):
    """Render n random (image | ground truth | prediction) rows."""
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    # A second dataset view without normalization, for displaying the image.
    raw = CamVidDataset(ds_root, ds.split,
                        transforms=get_transforms(ds.split, H, W, normalize=False))

    fig, axes = plt.subplots(len(idxs), 3, figsize=(13, 3.2 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        img_t, gt = ds[int(i)]
        disp_img, _ = raw[int(i)]
        x = img_t.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = model(x).argmax(1)[0].cpu().numpy()
        gt = np.asarray(gt)

        axes[row, 0].imshow(disp_img)
        axes[row, 0].set_title(raw.names[int(i)], fontsize=8)
        axes[row, 1].imshow(decode_segmap(gt))
        axes[row, 1].set_title("ground truth", fontsize=8)
        axes[row, 2].imshow(decode_segmap(pred))
        axes[row, 2].set_title("prediction", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    legend = [Patch(facecolor=PALETTE[i] / 255.0, label=n)
              for i, n in enumerate(CLASS_NAMES)]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    print(f"[eval] saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run name under outputs/runs/")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num-vis", type=int, default=6)
    p.add_argument("--out", default="outputs/runs")
    args = p.parse_args()

    if args.split == "test":
        print("\n*** WARNING: evaluating on TEST. Only do this on Day 8+, "
              "once per model, after all tuning is frozen. ***\n")

    global ds_root, H, W
    ds_root, H, W = args.root, args.height, args.width

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out) / args.run
    model, model_name, cfg = load_run_model(run_dir, device)

    ds = CamVidDataset(args.root, args.split,
                       transforms=get_transforms(args.split, H, W))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    res = evaluate(model, loader, device)
    print(f"\n[eval] {args.run} on {args.split}: "
          f"mIoU {res['miou']:.4f}  pixel_acc {res['pixel_acc']:.4f}")
    print(format_per_class(res["per_class_iou"]))

    out_json = run_dir / f"eval_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump({"run": args.run, "model": model_name,
                   "split": args.split, **res}, f, indent=2)
    print(f"[eval] wrote {out_json}")

    make_vis(model, ds, device, args.num_vis,
             run_dir / f"pred_{args.split}.png")


if __name__ == "__main__":
    main()
