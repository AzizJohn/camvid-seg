"""Boundary IoU: quantify how well predicted boundaries match ground-truth
boundaries, complementing standard (region) IoU.

Boundary IoU (Cheng et al., 2021) computes IoU only within a thin band of
width d pixels around object boundaries, so it is sensitive to edge quality
rather than to large region interiors. We report it per class and as a mean,
for each model, on the validation set. No retraining - uses best.pt.

The boundary band for a mask is obtained by eroding each class region and
subtracting the eroded region from the original, giving the set of pixels
within d of a boundary. Boundary IoU for a class is then the IoU of the
predicted and ground-truth boundary bands for that class.

Usage:
    python src/boundary_iou.py --runs unet_100 segformer_100 --split val --dilation 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from dataset import (
    CamVidDataset, get_transforms, NUM_CLASSES, VOID_INDEX, CLASS_NAMES,
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


def boundary_band(binary_mask, d):
    """Return the boundary band (pixels within d of the region boundary)
    of a binary uint8 mask, via erosion."""
    mask = binary_mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask  # empty
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
    eroded = cv2.erode(mask, kernel)
    return mask - eroded  # 1 on the band, 0 elsewhere


@torch.no_grad()
def accumulate(model, loader, device, d):
    """Accumulate boundary intersection and union per class over the set."""
    inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    union = np.zeros(NUM_CLASSES, dtype=np.float64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = model(images).argmax(1).cpu().numpy()
        gts = targets.numpy()
        for pred, gt in zip(preds, gts):
            valid = gt != VOID_INDEX
            for c in range(NUM_CLASSES):
                gt_c = (gt == c) & valid
                pr_c = (pred == c) & valid
                if gt_c.sum() == 0 and pr_c.sum() == 0:
                    continue
                gb = boundary_band(gt_c, d)
                pb = boundary_band(pr_c, d)
                # boundary IoU uses the intersection of the bands
                i = np.logical_and(gb, pb).sum()
                u = np.logical_or(gb, pb).sum()
                inter[c] += i
                union[c] += u
    return inter, union


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--dilation", type=int, default=3,
                   help="boundary band width d in pixels")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CamVidDataset(args.root, args.split,
                       transforms=get_transforms(args.split, args.height, args.width))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    results = {}
    for r in args.runs:
        model, mtype = load_model(Path(args.runs_dir) / r, device)
        inter, union = accumulate(model, loader, device, args.dilation)
        biou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        mean_biou = np.nanmean(biou)
        results[r] = (biou, mean_biou)
        print(f"[boundary] {r} ({mtype}): mean Boundary IoU = {mean_biou:.4f}")

    # ---- comparison table ---------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"boundary_iou_{args.split}.csv"
    with open(csv_path, "w") as f:
        f.write("class," + ",".join(args.runs) + "\n")
        for ci, c in enumerate(CLASS_NAMES):
            row = [c] + [f"{results[r][0][ci]:.4f}" for r in args.runs]
            f.write(",".join(row) + "\n")
        f.write("MEAN," + ",".join(f"{results[r][1]:.4f}" for r in args.runs) + "\n")
    print(f"[boundary] wrote {csv_path}")

    if len(args.runs) == 2:
        a, b = args.runs
        print(f"\n  Boundary IoU (d={args.dilation}px), {a} vs {b}:")
        print(f"    {'class':<12}{a:>14}{b:>14}{'delta':>9}")
        print("    " + "-" * 49)
        for ci, c in enumerate(CLASS_NAMES):
            va, vb = results[a][0][ci], results[b][0][ci]
            print(f"    {c:<12}{va:>14.3f}{vb:>14.3f}{vb - va:>+9.3f}")
        print("    " + "-" * 49)
        print(f"    {'MEAN':<12}{results[a][1]:>14.3f}{results[b][1]:>14.3f}"
              f"{results[b][1] - results[a][1]:>+9.3f}")


if __name__ == "__main__":
    main()
