"""Segmentation metrics: per-class IoU, mean IoU, pixel accuracy.

Wraps torchmetrics.JaccardIndex for IoU and computes pixel accuracy
manually. The void label (index 11) is excluded everywhere: predictions
on void pixels do not count, matching how the loss treats them.

Usage:
    meter = SegMetrics(device)
    meter.update(logits, target)   # call every batch
    results = meter.compute()      # dict at end of epoch
    meter.reset()                  # before the next epoch
"""

from typing import Dict

import torch
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import NUM_CLASSES, VOID_INDEX, CLASS_NAMES


class SegMetrics:
    def __init__(self, device: torch.device):
        self.device = device
        # average=None -> per-class IoU vector; ignore_index drops void.
        self.iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES,
            average=None,
            ignore_index=VOID_INDEX,
        ).to(device)
        self._correct = torch.tensor(0.0, device=device)
        self._total = torch.tensor(0.0, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        self.iou.update(preds, target)

        valid = target != VOID_INDEX
        self._correct += (preds[valid] == target[valid]).sum()
        self._total += valid.sum()

    @torch.no_grad()
    def compute(self) -> Dict:
        per_class = self.iou.compute()  # (NUM_CLASSES,)
        # torchmetrics returns nan for classes absent from all batches seen;
        # ignore those when averaging so mIoU stays meaningful on subsets.
        valid = ~torch.isnan(per_class)
        miou = per_class[valid].mean().item() if valid.any() else float("nan")
        pixel_acc = (self._correct / self._total.clamp(min=1)).item()
        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "per_class_iou": {
                CLASS_NAMES[i]: per_class[i].item()
                for i in range(NUM_CLASSES)
            },
        }

    def reset(self) -> None:
        self.iou.reset()
        self._correct.zero_()
        self._total.zero_()


def format_per_class(per_class_iou: Dict[str, float]) -> str:
    """One-line-per-class IoU table, sorted worst-first to spotlight the
    hard classes (which is where the CNN/Transformer story lives)."""
    items = sorted(per_class_iou.items(), key=lambda kv: kv[1])
    lines = ["  per-class IoU (worst first):"]
    for name, val in items:
        bar = "#" * int(max(val, 0) * 30)
        lines.append(f"    {name:<12} {val:6.3f}  {bar}")
    return "\n".join(lines)
