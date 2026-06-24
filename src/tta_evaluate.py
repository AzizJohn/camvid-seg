"""Test-time augmentation (TTA) evaluation.

Improves accuracy at inference with NO retraining by averaging predictions
over augmented views of each image, then comparing to the non-TTA baseline.
TTA is applied IDENTICALLY to both models, so the comparison stays fair.

Augmentations averaged (in probability space, after softmax):
  * horizontal flip (always)
  * multiple scales (optional, via --scales), each resized back before
    averaging

Also supports evaluating at a different base resolution than training via
--height/--width, which can recover thin classes (e.g. pole) that are
resolution-starved at 384x480.

Usage:
    # flip-only TTA at training resolution
    python src/tta_evaluate.py --run unet_100 --split val

    # flip + multi-scale TTA
    python src/tta_evaluate.py --run segformer_100 --split val \
        --scales 0.75 1.0 1.25

    # higher-resolution evaluation (still divisible by 32)
    python src/tta_evaluate.py --run unet_100 --split val \
        --height 736 --width 960
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CamVidDataset, get_transforms, NUM_CLASSES
from metrics import SegMetrics, format_per_class
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
def predict_proba(model, x, scales, use_flip, device):
    """Return averaged softmax probabilities (N,C,H,W) over TTA views.

    Each scaled view is resized to `scale`, run through the model, and the
    logits are resized back to the original size before softmax-averaging,
    so every view contributes at the same resolution.
    """
    H, W = x.shape[-2:]
    acc = torch.zeros(x.shape[0], NUM_CLASSES, H, W, device=device)
    n = 0
    for s in scales:
        if s == 1.0:
            xs = x
        else:
            h2 = int(round(H * s / 32)) * 32
            w2 = int(round(W * s / 32)) * 32
            xs = F.interpolate(x, size=(h2, w2), mode="bilinear",
                               align_corners=False)
        views = [xs]
        if use_flip:
            views.append(torch.flip(xs, dims=[-1]))
        for vi, v in enumerate(views):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(v)
            if vi == 1:  # undo the flip
                logits = torch.flip(logits, dims=[-1])
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)
            acc += F.softmax(logits, dim=1)
            n += 1
    return acc / n


@torch.no_grad()
def evaluate_tta(model, loader, scales, use_flip, device):
    m = SegMetrics(device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        proba = predict_proba(model, images, scales, use_flip, device)
        # SegMetrics expects logits; log of proba preserves the argmax and
        # is a valid stand-in for the metric (argmax of proba == argmax).
        m.update(proba, targets)
    return m.compute()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0],
                   help="scales to average, e.g. 0.75 1.0 1.25")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--runs-dir", default="outputs/runs")
    args = p.parse_args()

    if args.split == "test":
        print("\n*** TEST split: only on Day 8+, once per model. ***\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.runs_dir) / args.run
    model, mtype = load_model(run_dir, device)

    ds = CamVidDataset(args.root, args.split,
                       transforms=get_transforms(args.split, args.height, args.width))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    use_flip = not args.no_flip
    print(f"[tta] {args.run} ({mtype}) @ {args.height}x{args.width}, "
          f"scales={args.scales}, flip={use_flip}")
    res = evaluate_tta(model, loader, args.scales, use_flip, device)
    print(f"[tta] mIoU {res['miou']:.4f}  pixel_acc {res['pixel_acc']:.4f}")
    print(format_per_class(res["per_class_iou"]))

    # Save alongside the run for later aggregation.
    tag = f"tta_{args.height}x{args.width}_s{'_'.join(str(s) for s in args.scales)}"
    out_json = run_dir / f"eval_{args.split}_{tag}.json"
    with open(out_json, "w") as f:
        json.dump({"run": args.run, "model": mtype, "split": args.split,
                   "tta": {"scales": args.scales, "flip": use_flip,
                           "height": args.height, "width": args.width},
                   **res}, f, indent=2)
    print(f"[tta] wrote {out_json}")


if __name__ == "__main__":
    main()
