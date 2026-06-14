"""Aggregate all run results into comparison tables and the data-efficiency
plot. Reads every outputs/runs/<run>/eval_<split>.json it can find.

Produces:
  outputs/analysis/summary_<split>.csv     one row per run (mIoU, pixel acc,
                                            per-class IoU)
  outputs/analysis/per_class_<split>.csv    classes x runs matrix of IoU
  outputs/analysis/data_efficiency.png      mIoU vs data fraction, both models

Usage (from repo root, after running evaluate.py on the runs):
    python src/aggregate_results.py --split val
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES


def parse_run_name(name):
    """Extract (model, fraction) from names like 'unet_50' / 'segformer_100'."""
    m = re.match(r"(unet|segformer)_(\d+)$", name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect every eval json for this split.
    records = []
    for run_dir in sorted(runs_dir.iterdir()):
        ev = run_dir / f"eval_{args.split}.json"
        if ev.exists():
            with open(ev) as f:
                records.append(json.load(f))
    if not records:
        raise SystemExit(f"No eval_{args.split}.json found under {runs_dir}. "
                         "Run evaluate.py first.")
    print(f"Found {len(records)} evaluated runs for split '{args.split}'.")

    # ---- summary table (one row per run) ------------------------------
    summary_path = out_dir / f"summary_{args.split}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "miou", "pixel_acc"] + CLASS_NAMES)
        for r in records:
            pc = r["per_class_iou"]
            w.writerow(
                [r["run"], r["model"], f"{r['miou']:.4f}",
                 f"{r['pixel_acc']:.4f}"]
                + [f"{pc[c]:.4f}" for c in CLASS_NAMES]
            )
    print(f"  wrote {summary_path}")

    # ---- per-class matrix (classes x runs) ----------------------------
    per_class_path = out_dir / f"per_class_{args.split}.csv"
    run_names = [r["run"] for r in records]
    with open(per_class_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + run_names)
        for c in CLASS_NAMES:
            w.writerow([c] + [f"{r['per_class_iou'][c]:.4f}" for r in records])
    print(f"  wrote {per_class_path}")

    # ---- console comparison at 100% data ------------------------------
    full = {r["model"]: r for r in records
            if parse_run_name(r["run"]) and parse_run_name(r["run"])[1] == 100}
    if "unet" in full and "segformer" in full:
        u, s = full["unet"], full["segformer"]
        print(f"\n  Head-to-head at 100% data ({args.split}):")
        print(f"    {'class':<12}{'U-Net':>8}{'SegFormer':>11}{'Δ(S-U)':>9}")
        print("    " + "-" * 38)
        for c in CLASS_NAMES:
            du, ds = u["per_class_iou"][c], s["per_class_iou"][c]
            print(f"    {c:<12}{du:>8.3f}{ds:>11.3f}{ds - du:>+9.3f}")
        print("    " + "-" * 38)
        print(f"    {'mIoU':<12}{u['miou']:>8.3f}{s['miou']:>11.3f}"
              f"{s['miou'] - u['miou']:>+9.3f}")

    # ---- data-efficiency plot -----------------------------------------
    curves = {}  # model -> {fraction: miou}
    for r in records:
        parsed = parse_run_name(r["run"])
        if parsed:
            model, frac = parsed
            curves.setdefault(model, {})[frac] = r["miou"]

    if curves:
        plt.figure(figsize=(7, 5))
        for model, pts in sorted(curves.items()):
            fracs = sorted(pts)
            mious = [pts[f] for f in fracs]
            plt.plot(fracs, mious, marker="o", linewidth=2, label=model)
            for fr, mi in zip(fracs, mious):
                plt.annotate(f"{mi:.3f}", (fr, mi),
                             textcoords="offset points", xytext=(0, 8),
                             fontsize=8, ha="center")
        plt.xlabel("training data (%)")
        plt.ylabel(f"{args.split} mIoU")
        plt.title("Data efficiency: CNN vs Transformer on CamVid")
        plt.xticks([25, 50, 100])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / "data_efficiency.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  wrote {plot_path}")

    print("\nDone. These tables and the plot go straight into the report.")


if __name__ == "__main__":
    main()
