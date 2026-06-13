"""Generate fixed training subsets for the data-efficiency experiments.

Writes outputs/subsets/train_{25,50,100}.txt, each a list of image
filenames. The subsets are nested and deterministic (fixed seed), and the
SAME files are used for both U-Net and SegFormer so the comparison at each
data fraction is fair. Val and test are never subsampled.

Usage (from repo root):
    python src/make_subsets.py --root data/CamVid --seed 42
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/CamVid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/subsets")
    args = parser.parse_args()

    train_dir = Path(args.root) / "train"
    names = sorted(p.name for p in train_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No training images in {train_dir}")
    n = len(names)
    print(f"Found {n} training images.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)  # one shuffle; subsets are prefixes of it

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for frac in (25, 50, 100):
        k = max(1, round(n * frac / 100))
        chosen = sorted(names[i] for i in perm[:k])  # nested: 25 ⊂ 50 ⊂ 100
        path = out_dir / f"train_{frac}.txt"
        path.write_text("\n".join(chosen) + "\n")
        print(f"  {frac:>3}% -> {k:>3} images -> {path}")

    print(
        f"\nDone (seed={args.seed}). These lists are consumed by train.py "
        "via --subset; both models must use the same files."
    )


if __name__ == "__main__":
    main()
