"""Shared training script for both U-Net and SegFormer.

The SAME script trains both models - only --model and a few hyperparameters
differ. This is what makes the comparison fair: identical data pipeline,
loss, metrics, schedule and logging for both architectures.

Examples
--------
Headline U-Net run (100% data):
    python src/train.py --model unet --epochs 120 --lr 3e-4 \
        --batch-size 8 --run-name unet_100

Headline SegFormer run (100% data):
    python src/train.py --model segformer --epochs 120 --lr 6e-5 \
        --batch-size 8 --run-name segformer_100

Data-efficiency run (50% subset):
    python src/train.py --model unet --epochs 120 --lr 3e-4 \
        --subset outputs/subsets/train_50.txt --run-name unet_50

Outputs per run (under outputs/runs/<run-name>/):
    best.pt        checkpoint with the highest val mIoU
    last.pt        most recent checkpoint
    metrics.csv    per-epoch train loss, val mIoU, val pixel acc
    config.json    the exact arguments used (for reproducibility)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CamVidDataset, get_transforms, NUM_CLASSES
from losses import CombinedLoss, load_class_weights
from metrics import SegMetrics, format_per_class
from models import build_model, count_parameters


def read_subset(path):
    if path is None:
        return None
    names = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    print(f"[data] subset: {len(names)} training images from {path}")
    return names


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    train_files = read_subset(args.subset)
    train_ds = CamVidDataset(
        args.root, "train",
        transforms=get_transforms("train", args.height, args.width),
        file_list=train_files,
    )
    val_ds = CamVidDataset(
        args.root, "val",
        transforms=get_transforms("val", args.height, args.width),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["unet", "segformer"])
    p.add_argument("--root", default="data/CamVid")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subset", default=None,
                   help="path to a train_xx.txt file (data-efficiency runs)")
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable median-frequency CE weighting")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (AMP is on by default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out", default="outputs/runs")
    p.add_argument("--overfit", type=int, default=0,
                   help="if >0, train on this many images with no aug "
                        "(sanity check; should reach ~1.0 train mIoU)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.model}_{int(time.time())}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[run] {run_name} on {device}")

    # --- data -----------------------------------------------------------
    if args.overfit > 0:
        # Tiny fixed subset, validation == training set, no augmentation:
        # the loop must memorize these images or something is broken.
        from dataset import CamVidDataset as DS
        full = DS(args.root, "train",
                  transforms=get_transforms("val", args.height, args.width))
        names = full.names[: args.overfit]
        train_ds = DS(args.root, "train",
                      transforms=get_transforms("val", args.height, args.width),
                      file_list=names)
        train_loader = DataLoader(train_ds, batch_size=min(args.batch_size,
                                  args.overfit), shuffle=True,
                                  num_workers=2, drop_last=False)
        val_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)
        print(f"[overfit] {len(names)} images, val==train")
    else:
        train_loader, val_loader = build_loaders(args)

    # --- model / loss / optim ------------------------------------------
    model = build_model(args.model).to(device)
    print(f"[model] {args.model}: {count_parameters(model):,} params")

    class_weights = None if (args.no_class_weights or args.overfit > 0) \
        else load_class_weights(device=device)
    criterion = CombinedLoss(
        class_weights=class_weights,
        ce_weight=args.ce_weight, dice_weight=args.dice_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Poly LR decay (standard for segmentation).
    def poly(epoch):
        return (1 - epoch / max(args.epochs, 1)) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly)

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[amp] mixed precision: {use_amp}")

    metrics = SegMetrics(device)
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_miou", "val_pixel_acc", "lr"]
        )

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                metrics.update(logits, targets)
        res = metrics.compute()
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{res['miou']:.4f}",
                 f"{res['pixel_acc']:.4f}", f"{lr_now:.2e}"]
            )

        msg = (f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
               f"val mIoU {res['miou']:.4f}  pix {res['pixel_acc']:.4f}")
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "miou": res["miou"], "args": vars(args)},
                   run_dir / "last.pt")
        if res["miou"] > best_miou:
            best_miou = res["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "miou": res["miou"], "args": vars(args)},
                       run_dir / "best.pt")
            msg += "  <- best"
        print(msg)
        if epoch == args.epochs or epoch % 20 == 0:
            print(format_per_class(res["per_class_iou"]))

    print(f"\n[done] best val mIoU = {best_miou:.4f}  ({run_dir}/best.pt)")
    if args.overfit > 0:
        ok = best_miou > 0.95
        print(f"[overfit] {'PASS' if ok else 'FAIL'}: "
              f"best mIoU {best_miou:.3f} "
              f"({'>' if ok else '<='} 0.95 threshold)")


if __name__ == "__main__":
    main()
