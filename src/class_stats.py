"""Day 1: per-class pixel statistics and class weights for CamVid.

Computes, over the training masks:
  * pixel count and share per class (this powers the "performance vs.
    object size" analysis later in the project),
  * median-frequency-balancing weights (Eigen & Fergus, 2015):
        freq(c)   = pixels of class c / total pixels in images containing c
        weight(c) = median(freq) / freq(c)
    These can be passed to CrossEntropyLoss to counter class imbalance.

Usage (from the repo root):
    python src/class_stats.py --root data/CamVid

Writes outputs/class_stats.json and prints a summary table.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import CLASS_NAMES, NUM_CLASSES, VOID_INDEX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/CamVid")
    parser.add_argument("--out", default="outputs/class_stats.json")
    args = parser.parse_args()

    ann_dir = Path(args.root) / "trainannot"
    mask_paths = sorted(ann_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(
            f"No masks found in {ann_dir}. Run scripts/download_camvid.sh."
        )
    print(f"Scanning {len(mask_paths)} training masks...")

    n_bins = VOID_INDEX + 1  # classes 0..10 plus void = 12 bins
    pixel_counts = np.zeros(n_bins, dtype=np.int64)
    # For median frequency balancing: total pixels of all images in which
    # class c appears at least once.
    pixels_where_present = np.zeros(n_bins, dtype=np.int64)
    image_count = np.zeros(n_bins, dtype=np.int64)

    for p in mask_paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Failed to read {p}"
        assert mask.max() <= VOID_INDEX, f"{p.name}: labels > {VOID_INDEX}"

        counts = np.bincount(mask.ravel(), minlength=n_bins)
        pixel_counts += counts
        present = counts > 0
        pixels_where_present[present] += mask.size
        image_count[present] += 1

    total = int(pixel_counts.sum())
    share = pixel_counts / total

    # Median frequency balancing over the 11 real classes (void excluded).
    freq = pixel_counts[:NUM_CLASSES] / np.maximum(
        pixels_where_present[:NUM_CLASSES], 1
    )
    weights = np.median(freq) / np.maximum(freq, 1e-12)

    header = f"{'class':<12}{'pixels':>14}{'share %':>9}{'images':>8}{'weight':>9}"
    print("\n" + header)
    print("-" * len(header))
    order = np.argsort(-pixel_counts[:NUM_CLASSES])
    for c in order:
        print(
            f"{CLASS_NAMES[c]:<12}{pixel_counts[c]:>14,}"
            f"{100 * share[c]:>8.2f}%{image_count[c]:>8}"
            f"{weights[c]:>9.3f}"
        )
    print(
        f"{'void':<12}{pixel_counts[VOID_INDEX]:>14,}"
        f"{100 * share[VOID_INDEX]:>8.2f}%{image_count[VOID_INDEX]:>8}"
        f"{'ignored':>9}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "class_names": CLASS_NAMES,
                "pixel_counts": pixel_counts[:NUM_CLASSES].tolist(),
                "void_pixel_count": int(pixel_counts[VOID_INDEX]),
                "pixel_share": share[:NUM_CLASSES].tolist(),
                "images_containing_class": image_count[:NUM_CLASSES].tolist(),
                "median_freq_weights": weights.tolist(),
            },
            f,
            indent=2,
        )
    print(f"\nSaved {out_path}")
    print(
        "Note the imbalance: road/sky/building dominate while pole, "
        "sign_symbol, pedestrian and bicyclist are tiny - keep this table "
        "for the per-class analysis in the report."
    )


if __name__ == "__main__":
    main()
