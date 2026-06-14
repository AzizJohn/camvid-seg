"""Side-by-side qualitative comparison of both models on the same images.

Renders rows of: input | ground truth | U-Net pred | SegFormer pred.
This is the report's centerpiece figure - it makes the CNN-vs-Transformer
difference visible on the hard classes (poles, signs, distant people).

Usage (after both models' best.pt exist):
    python src/compare_models.py --unet-run unet_100 \
        --segformer-run segformer_100 --split val --num 6

Optionally focus on images where the two models disagree most (most
informative for the discussion):
    python src/compare_models.py --unet-run unet_100 \
        --segformer-run segformer_100 --split val --num 6 --mode disagree
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
    CLASS_NAMES, PALETTE, VOID_INDEX,
)
from models import build_model


def load_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg["model"]


@torch.no_grad()
def predict(model, x, device):
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        return model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unet-run", required=True)
    p.add_argument("--segformer-run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--mode", default="random", choices=["random", "disagree"])
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = Path(args.runs_dir)
    unet, _ = load_model(runs / args.unet_run, device)
    seg, _ = load_model(runs / args.segformer_run, device)

    norm_ds = CamVidDataset(args.root, args.split,
                            transforms=get_transforms(args.split, args.height, args.width))
    raw_ds = CamVidDataset(args.root, args.split,
                           transforms=get_transforms(args.split, args.height, args.width, normalize=False))

    # Choose which images to show.
    if args.mode == "disagree":
        # Rank images by how much the two models' predictions differ on
        # valid pixels - these are the most informative for the discussion.
        scores = []
        for i in range(len(norm_ds)):
            img, gt = norm_ds[i]
            gt = np.asarray(gt)
            pu = predict(unet, img, device)
            ps = predict(seg, img, device)
            valid = gt != VOID_INDEX
            disagree = ((pu != ps) & valid).sum() / max(valid.sum(), 1)
            scores.append((disagree, i))
        scores.sort(reverse=True)
        idxs = [i for _, i in scores[:args.num]]
        print(f"[compare] most-disagreeing images: {idxs}")
    else:
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(norm_ds), size=min(args.num, len(norm_ds)),
                          replace=False).tolist()

    titles = ["input", "ground truth", "U-Net", "SegFormer"]
    fig, axes = plt.subplots(len(idxs), 4, figsize=(17, 3.2 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        img, gt = norm_ds[i]
        disp, _ = raw_ds[i]
        gt = np.asarray(gt)
        pu = predict(unet, img, device)
        ps = predict(seg, img, device)
        panels = [disp, decode_segmap(gt), decode_segmap(pu), decode_segmap(ps)]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel)
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(raw_ds.names[i], fontsize=7)

    legend = [Patch(facecolor=PALETTE[k] / 255.0, label=n)
              for k, n in enumerate(CLASS_NAMES)]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{args.mode}_{args.split}.png"
    fig.savefig(out_path, dpi=150)
    print(f"[compare] saved {out_path}")


if __name__ == "__main__":
    main()
