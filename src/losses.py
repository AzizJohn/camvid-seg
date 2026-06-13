"""Loss functions for CamVid segmentation.

Combined loss = CE_weight * CrossEntropy + dice_weight * Dice.

Both terms ignore the void label (index 11). The CrossEntropy term uses the
median-frequency class weights from class_stats.json so rare classes
(bicyclist, pedestrian, pole) are not drowned out by road/sky/building.
The Dice term improves overlap on small structures and is robust to
imbalance. Using both is standard practice for imbalanced segmentation.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES, VOID_INDEX


def load_class_weights(
    stats_path: str = "outputs/class_stats.json",
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Load median-frequency class weights produced by class_stats.py."""
    p = Path(stats_path)
    if not p.exists():
        print(f"[loss] {stats_path} not found - using uniform CE weights.")
        return None
    with open(p) as f:
        stats = json.load(f)
    weights = torch.tensor(stats["median_freq_weights"], dtype=torch.float32)
    assert weights.numel() == NUM_CLASSES, "weight/class count mismatch"
    if device is not None:
        weights = weights.to(device)
    return weights


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss, averaged over classes, ignoring void.

    Pixels labelled void are masked out of both the prediction and the
    target before computing the per-class overlap.
    """

    def __init__(self, ignore_index: int = VOID_INDEX, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); target: (N, H, W) with values 0..C plus void.
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid = (target != self.ignore_index)  # (N, H, W) bool
        # Clamp void to a real index so one_hot doesn't choke, then zero it
        # out with the validity mask.
        target_clamped = target.clone()
        target_clamped[~valid] = 0
        target_oh = F.one_hot(target_clamped, num_classes)  # (N,H,W,C)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()  # (N,1,H,W)
        probs = probs * valid
        target_oh = target_oh * valid

        dims = (0, 2, 3)  # sum over batch and spatial -> per-class scores
        intersection = (probs * target_oh).sum(dims)
        cardinality = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """ce_weight * weighted-CE + dice_weight * Dice, both ignoring void."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = VOID_INDEX,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.ce(logits, target)
            + self.dice_weight * self.dice(logits, target)
        )
