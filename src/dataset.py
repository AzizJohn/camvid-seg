"""CamVid dataset (11-class version from the SegNet-Tutorial repository).

Expected directory layout (created by scripts/download_camvid.sh):

    data/CamVid/
        train/        367 RGB images, 360x480, .png
        trainannot/   367 label masks, single-channel .png, values 0..11
        val/          101 images
        valannot/
        test/         233 images
        testannot/

Label encoding: 0..10 are the 11 classes below, 11 is the void/unlabeled
class, which must be ignored in the loss and in all metrics.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "sky",          # 0
    "building",     # 1
    "pole",         # 2
    "road",         # 3
    "pavement",     # 4
    "tree",         # 5
    "sign_symbol",  # 6
    "fence",        # 7
    "car",          # 8
    "pedestrian",   # 9
    "bicyclist",    # 10
]

NUM_CLASSES = len(CLASS_NAMES)  # 11
VOID_INDEX = 11                 # ignore_index for loss and metrics

# Standard CamVid color palette (RGB), index 11 = void (black).
PALETTE = np.array(
    [
        [128, 128, 128],  # sky
        [128, 0, 0],      # building
        [192, 192, 128],  # pole
        [128, 64, 128],   # road
        [0, 0, 192],      # pavement
        [128, 128, 0],    # tree
        [192, 128, 128],  # sign_symbol
        [64, 64, 128],    # fence
        [64, 0, 128],     # car
        [64, 64, 0],      # pedestrian
        [0, 128, 192],    # bicyclist
        [0, 0, 0],        # void
    ],
    dtype=np.uint8,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """Convert an HxW index mask (values 0..11) to an HxWx3 RGB image."""
    return PALETTE[mask]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    split: str,
    height: int = 384,
    width: int = 480,
    normalize: bool = True,
) -> A.Compose:
    """Albumentations pipeline shared by both models (fair comparison).

    CamVid images are 360x480. Both U-Net (ResNet) and SegFormer require
    spatial dims divisible by 32, so we pad 360 -> 384 with the void label,
    which is then ignored by the loss and the metrics.

    With normalize=False the pipeline returns raw uint8 numpy arrays,
    which is useful for visual sanity checks.
    """
    pad = A.PadIfNeeded(
        min_height=height,
        min_width=width,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        mask_value=VOID_INDEX,
    )

    if split == "train":
        aug = [
            pad,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
        ]
    elif split in ("val", "test"):
        aug = [pad]
    else:
        raise ValueError(f"Unknown split: {split!r}")

    if normalize:
        aug += [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    return A.Compose(aug)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CamVidDataset(Dataset):
    """CamVid semantic segmentation dataset (11 classes + void).

    Args:
        root: path to data/CamVid (the SegNet-Tutorial layout).
        split: "train", "val" or "test".
        transforms: an albumentations Compose (see get_transforms).
        file_list: optional list of image filenames; restricts the dataset
            to this subset. Used on Day 2+ for the 25%/50% data-efficiency
            experiments so subsets stay fixed across both models.
    """

    SPLIT_DIRS = {
        "train": ("train", "trainannot"),
        "val": ("val", "valannot"),
        "test": ("test", "testannot"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        file_list: Optional[List[str]] = None,
    ):
        if split not in self.SPLIT_DIRS:
            raise ValueError(f"Unknown split: {split!r}")

        img_sub, ann_sub = self.SPLIT_DIRS[split]
        self.img_dir = Path(root) / img_sub
        self.ann_dir = Path(root) / ann_sub
        if not self.img_dir.is_dir() or not self.ann_dir.is_dir():
            raise FileNotFoundError(
                f"CamVid not found under {root!r}. "
                "Run scripts/download_camvid.sh first."
            )

        names = sorted(p.name for p in self.img_dir.glob("*.png"))
        if file_list is not None:
            wanted = set(file_list)
            names = [n for n in names if n in wanted]
            missing = wanted - set(names)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} files from file_list not found, "
                    f"e.g. {sorted(missing)[:3]}"
                )
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.names = names
        self.split = split
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]

        image = cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR)
        assert image is not None, f"Failed to read image {name}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.ann_dir / name), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Failed to read mask {name}"

        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # After ToTensorV2 the mask is a torch tensor; the loss expects long.
        if hasattr(mask, "long"):
            mask = mask.long()

        return image, mask
