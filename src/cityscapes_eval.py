"""Cross-dataset generalization: evaluate CamVid-trained models on Cityscapes.

NO retraining. We run the existing CamVid-trained best.pt checkpoints on
Cityscapes validation images (a different city, camera, resolution and
annotation style) and measure how much each model degrades. This tests
which architecture generalizes better under domain shift.

The two datasets use different label sets, so we map Cityscapes' labels
onto CamVid's 11 classes (see CS_TO_CAMVID). Cityscapes classes with no
CamVid equivalent are mapped to void and ignored in the metric. Predictions
stay in CamVid label space; only the Cityscapes ground truth is remapped.

Expected Cityscapes layout (official):
    cityscapes/
        leftImg8bit/val/<city>/<...>_leftImg8bit.png
        gtFine/val/<city>/<...>_gtFine_labelIds.png

Usage:
    python src/cityscapes_eval.py --runs unet_100 segformer_100 \
        --cs-root data/cityscapes --num-vis 6
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import (
    decode_segmap, CLASS_NAMES, PALETTE, NUM_CLASSES, VOID_INDEX,
    IMAGENET_MEAN, IMAGENET_STD,
)
from metrics import SegMetrics, format_per_class
from models import build_model

# ---------------------------------------------------------------------------
# Cityscapes labelIds (0..33) -> CamVid class index (0..10), or VOID to ignore.
# CamVid: 0 sky,1 building,2 pole,3 road,4 pavement,5 tree,6 sign_symbol,
#         7 fence,8 car,9 pedestrian,10 bicyclist
# Cityscapes labelIds reference (trainId scheme differs; we use raw labelIds):
#   7 road, 8 sidewalk, 11 building, 12 wall, 13 fence, 17 pole,
#   19 traffic light, 20 traffic sign, 21 vegetation, 22 terrain, 23 sky,
#   24 person, 25 rider, 26 car, 27 truck, 28 bus, 31 train,
#   32 motorcycle, 33 bicycle
# ---------------------------------------------------------------------------
V = VOID_INDEX
CS_TO_CAMVID = {
    7: 3,    # road -> road
    8: 4,    # sidewalk -> pavement
    9: 3,    # parking -> road (approx)
    11: 1,   # building -> building
    12: 1,   # wall -> building (approx)
    13: 7,   # fence -> fence
    17: 2,   # pole -> pole
    18: 2,   # polegroup -> pole
    19: 6,   # traffic light -> sign_symbol (approx)
    20: 6,   # traffic sign -> sign_symbol
    21: 5,   # vegetation -> tree
    23: 0,   # sky -> sky
    24: 9,   # person -> pedestrian
    25: 10,  # rider -> bicyclist (approx)
    26: 8,   # car -> car
    27: 8,   # truck -> car (approx)
    28: 8,   # bus -> car (approx)
    32: 10,  # motorcycle -> bicyclist (approx)
    33: 10,  # bicycle -> bicyclist
}


def remap_cityscapes(label_ids):
    """Map a Cityscapes labelIds mask to CamVid class indices; unmapped->void."""
    out = np.full(label_ids.shape, VOID_INDEX, dtype=np.uint8)
    for cs_id, cam_id in CS_TO_CAMVID.items():
        out[label_ids == cs_id] = cam_id
    return out


class CityscapesAsCamVid(Dataset):
    """Cityscapes val images with labels remapped to CamVid's 11 classes.

    Images are resized to a CamVid-like resolution so the domain shift is
    in content/style, not wildly different scale. Default 384x768 keeps the
    2:1 Cityscapes aspect ratio and is divisible by 32.
    """

    def __init__(self, cs_root, height=384, width=768):
        cs_root = Path(cs_root)
        self.img_dir = cs_root / "leftImg8bit" / "val"
        self.ann_dir = cs_root / "gtFine" / "val"
        self.images = sorted(self.img_dir.rglob("*_leftImg8bit.png"))
        if not self.images:
            raise FileNotFoundError(
                f"No Cityscapes val images under {self.img_dir}. "
                "Check --cs-root and that leftImg8bit/val is extracted."
            )
        self.height, self.width = height, width
        self.norm = A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.resize_mask = A.Resize(height, width,
                                    interpolation=cv2.INTER_NEAREST)

    def gt_path_for(self, img_path):
        # .../val/<city>/<name>_leftImg8bit.png -> gtFine .../<name>_gtFine_labelIds.png
        city = img_path.parent.name
        name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        return self.ann_dir / city / name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        label_ids = cv2.imread(str(self.gt_path_for(img_path)),
                               cv2.IMREAD_GRAYSCALE)
        mask = remap_cityscapes(label_ids)
        mask = self.resize_mask(image=mask)["image"]
        out = self.norm(image=image)
        return out["image"], torch.from_numpy(mask).long(), str(img_path.name)


def load_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg["model"]


@torch.no_grad()
def evaluate(model, loader, device):
    m = SegMetrics(device)
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        m.update(logits, masks)
    return m.compute()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cs-root", default="data/cityscapes")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num-vis", type=int, default=6)
    p.add_argument("--runs-dir", default="outputs/runs")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CityscapesAsCamVid(args.cs_root, args.height, args.width)
    print(f"[cityscapes] {len(ds)} val images, remapped to CamVid classes")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for r in args.runs:
        model, mtype = load_model(Path(args.runs_dir) / r, device)
        res = evaluate(model, loader, device)
        results[r] = res
        print(f"\n[cityscapes] {r} ({mtype}): mIoU {res['miou']:.4f}  "
              f"pixel_acc {res['pixel_acc']:.4f}")
        print(format_per_class(res["per_class_iou"]))
        with open(Path(args.runs_dir) / r / "eval_cityscapes.json", "w") as f:
            json.dump({"run": r, "model": mtype, "dataset": "cityscapes_val",
                       **res}, f, indent=2)

    # comparison summary
    csv_path = out_dir / "cityscapes_summary.csv"
    with open(csv_path, "w") as f:
        f.write("run,miou,pixel_acc," + ",".join(CLASS_NAMES) + "\n")
        for r in args.runs:
            res = results[r]
            f.write(f"{r},{res['miou']:.4f},{res['pixel_acc']:.4f}," +
                    ",".join(f"{res['per_class_iou'][c]:.4f}"
                             for c in CLASS_NAMES) + "\n")
    print(f"\n[cityscapes] wrote {csv_path}")

    # qualitative panel for the first model vs second
    if len(args.runs) >= 1:
        models = {r: load_model(Path(args.runs_dir) / r, device)[0]
                  for r in args.runs}
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(ds), size=min(args.num_vis, len(ds)),
                          replace=False).tolist()
        ncol = 2 + len(args.runs)
        fig, axes = plt.subplots(len(idxs), ncol,
                                 figsize=(4.2 * ncol, 2.6 * len(idxs)))
        if len(idxs) == 1:
            axes = axes[None, :]
        # raw images for display
        disp_norm = A.Resize(args.height, args.width)
        for row, i in enumerate(idxs):
            x, mask, name = ds[i]
            img_path = ds.images[i]
            disp = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            disp = disp_norm(image=disp)["image"]
            axes[row, 0].imshow(disp)
            axes[row, 1].imshow(decode_segmap(mask.numpy()))
            if row == 0:
                axes[row, 0].set_title("Cityscapes input", fontsize=10)
                axes[row, 1].set_title("GT (remapped)", fontsize=10)
            for col, r in enumerate(args.runs):
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=device.type == "cuda"):
                    pred = models[r](x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
                axes[row, 2 + col].imshow(decode_segmap(pred))
                if row == 0:
                    axes[row, 2 + col].set_title(r, fontsize=10)
            for ax in axes[row]:
                ax.axis("off")
        legend = [Patch(facecolor=PALETTE[k] / 255.0, label=n)
                  for k, n in enumerate(CLASS_NAMES)]
        fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        vis_path = out_dir / "cityscapes_qualitative.png"
        fig.savefig(vis_path, dpi=150)
        print(f"[cityscapes] wrote {vis_path}")


if __name__ == "__main__":
    main()
