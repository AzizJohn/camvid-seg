"""Day 1 sanity check: visualize CamVid samples and validate the labels.

Renders a grid of (image | colored mask | overlay) rows plus a class-color
legend, and asserts that every mask only contains values in [0, 11].

Usage (from the repo root):
    python src/visualize_data.py --root data/CamVid --split train --num 6
    python src/visualize_data.py --root data/CamVid --split val

Output is saved to outputs/data_check_<split>.png (no display needed,
works over SSH).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we only save figures, never show them
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from dataset import (
    CamVidDataset,
    get_transforms,
    decode_segmap,
    CLASS_NAMES,
    PALETTE,
    VOID_INDEX,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/CamVid")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=6,
                        help="number of samples to draw")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    # normalize=False -> raw uint8 numpy arrays, ideal for visualization.
    # For the train split this also shows the augmentations in action.
    ds = CamVidDataset(
        args.root, args.split,
        transforms=get_transforms(args.split, normalize=False),
    )
    print(f"Split '{args.split}': {len(ds)} samples")

    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False)

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, mask = ds[int(idx)]
        mask = np.asarray(mask)

        # --- label sanity checks -------------------------------------
        uniq = np.unique(mask)
        assert uniq.min() >= 0 and uniq.max() <= VOID_INDEX, (
            f"Sample {ds.names[int(idx)]} has labels outside [0, "
            f"{VOID_INDEX}]: {uniq}"
        )

        color_mask = decode_segmap(mask)
        overlay = (0.6 * image + 0.4 * color_mask).astype(np.uint8)

        present = ", ".join(
            CLASS_NAMES[c] if c < len(CLASS_NAMES) else "void"
            for c in uniq
        )
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(ds.names[int(idx)], fontsize=8)
        axes[row, 1].imshow(color_mask)
        axes[row, 1].set_title(f"labels: {present}", fontsize=6)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("overlay", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    legend = [
        Patch(facecolor=PALETTE[i] / 255.0, label=name)
        for i, name in enumerate(CLASS_NAMES)
    ] + [Patch(facecolor=PALETTE[VOID_INDEX] / 255.0, label="void (ignored)")]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_check_{args.split}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print("All label sanity checks passed.")


if __name__ == "__main__":
    main()
