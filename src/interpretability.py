"""Interpretability visuals: where does each model 'look'?

Renders, on the same frames, a heatmap of encoder feature activation for
each model, overlaid on the input:
  * U-Net (CNN): Grad-CAM-style map from the last encoder stage.
  * SegFormer (Transformer): mean activation magnitude of the last encoder
    stage (a proxy for attention focus).

IMPORTANT CAVEAT (state this in the report): the two heatmaps come from
different mechanisms and are NOT directly comparable as rigorous evidence.
They are qualitative intuition for how a local-receptive-field model and a
global-attention model distribute their focus. Use them as illustration,
not proof.

Usage:
    python src/interpretability.py --unet-run unet_100 \
        --segformer-run segformer_100 --split val --num 4
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CamVidDataset, get_transforms
from models import build_model


def load(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg["model"]


def normalize_map(m):
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def unet_cam(model, x):
    """Activation magnitude of the deepest U-Net encoder features.
    smp U-Net exposes model.encoder; we take the last feature map and use
    the per-pixel L2 norm across channels as a focus proxy."""
    feats = model.encoder(x)          # list of feature maps, coarse..fine
    deep = feats[-1]                  # deepest (smallest spatial) map
    cam = deep.pow(2).sum(1).sqrt()[0] # (h,w) L2 norm over channels
    cam = F.interpolate(cam[None, None], size=x.shape[-2:],
                        mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(cam.detach().cpu().numpy())


def segformer_cam(model, x):
    """Activation magnitude of SegFormer's deepest encoder stage.

    We request hidden states from the full model, which returns 4D feature
    maps (B, C, h, w) for each stage - avoiding any assumption that the
    token count is a perfect square (CamVid maps are non-square, e.g. 12x15).
    """
    out = model.net(pixel_values=x, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if not hs:
        raise RuntimeError("SegFormer did not return hidden_states")
    deep = hs[-1]                     # deepest stage

    if deep.dim() == 3:
        # Fallback only: (B, N, C). Recover h,w from the input stride (/32
        # at the last stage) rather than assuming a square.
        B, N, C = deep.shape
        h = max(1, x.shape[-2] // 32)
        w = max(1, N // h)
        deep = deep.transpose(1, 2).reshape(B, C, h, w)
    cam = deep.pow(2).sum(1).sqrt()[0]
    cam = F.interpolate(cam[None, None], size=x.shape[-2:],
                        mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(cam.detach().cpu().numpy())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unet-run", required=True)
    p.add_argument("--segformer-run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = Path(args.runs_dir)
    unet, _ = load(runs / args.unet_run, device)
    seg, _ = load(runs / args.segformer_run, device)

    norm_ds = CamVidDataset(args.root, args.split,
                            transforms=get_transforms(args.split, args.height, args.width))
    raw_ds = CamVidDataset(args.root, args.split,
                           transforms=get_transforms(args.split, args.height, args.width, normalize=False))

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(norm_ds), size=min(args.num, len(norm_ds)),
                      replace=False).tolist()

    titles = ["input", "U-Net focus", "SegFormer focus"]
    fig, axes = plt.subplots(len(idxs), 3, figsize=(13, 3.4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        x, _ = norm_ds[i]
        disp, _ = raw_ds[i]
        xb = x.unsqueeze(0).to(device)
        try:
            cu = unet_cam(unet, xb)
        except Exception as e:
            print(f"[interp] U-Net CAM failed: {e}"); cu = np.zeros(disp.shape[:2])
        try:
            cs = segformer_cam(seg, xb)
        except Exception as e:
            print(f"[interp] SegFormer CAM failed: {e}"); cs = np.zeros(disp.shape[:2])

        axes[row, 0].imshow(disp)
        axes[row, 1].imshow(disp); axes[row, 1].imshow(cu, cmap="jet", alpha=0.5)
        axes[row, 2].imshow(disp); axes[row, 2].imshow(cs, cmap="jet", alpha=0.5)
        for col in range(3):
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=11)
            axes[row, col].axis("off")

    fig.suptitle("Encoder focus: local (U-Net) vs global (SegFormer) "
                 "— qualitative, not directly comparable", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"interpretability_{args.split}.png"
    fig.savefig(out_path, dpi=150)
    print(f"[interp] saved {out_path}")


if __name__ == "__main__":
    main()
