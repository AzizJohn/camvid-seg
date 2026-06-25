"""
ADD THIS TO losses.py
=====================

Paste the FocalLoss class and CombinedFocalLoss class below into your existing
src/losses.py (after the DiceLoss class). They reuse the same void-ignoring
and class-weighting conventions as your CE+Dice loss, so the comparison stays
controlled: the ONLY thing that changes versus baseline is CE -> Focal.

Then add the small block shown at the bottom to train.py so a --loss flag
selects between the CE+Dice baseline and the new Focal+Dice variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes these are already imported at the top of losses.py:
#   from dataset import NUM_CLASSES, VOID_INDEX


# ===========================================================================
# 1) Paste this class into losses.py
# ===========================================================================
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), ignoring the void label.

    Focal loss = - alpha_c * (1 - p_t)^gamma * log(p_t), summed over valid
    pixels. The (1 - p_t)^gamma factor down-weights well-classified pixels
    so the gradient concentrates on hard pixels (e.g. thin-structure
    boundaries like poles and signs).

    alpha (the per-class weight vector) plays the same role as the
    median-frequency class weights in the CE baseline, so rare classes are
    still up-weighted; gamma controls how aggressively easy pixels are
    down-weighted (gamma=2 is the standard default; gamma=0 recovers
    weighted cross-entropy).
    """

    def __init__(self, alpha=None, gamma: float = 2.0,
                 ignore_index: int = None):
        super().__init__()
        from dataset import VOID_INDEX
        self.gamma = gamma
        self.ignore_index = VOID_INDEX if ignore_index is None else ignore_index
        # alpha: optional (NUM_CLASSES,) tensor of per-class weights.
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); target: (N, H, W)
        # Standard CE per pixel (no reduction), with the same class weights
        # and ignore_index as the baseline; then apply the focal modulation.
        ce = F.cross_entropy(
            logits, target,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (N, H, W); ignored pixels contribute 0

        # p_t = exp(-ce_unweighted); but ce already includes the class weight.
        # Recover the unweighted probability for the focal term:
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=1)
            # gather the log-prob of the target class per pixel
            valid = target != self.ignore_index
            safe_t = target.clone()
            safe_t[~valid] = 0
            pt = logp.gather(1, safe_t.unsqueeze(1)).squeeze(1).exp()  # (N,H,W)
            focal_factor = (1.0 - pt).clamp(min=0).pow(self.gamma)
            focal_factor[~valid] = 0.0

        loss = focal_factor * ce
        denom = valid.sum().clamp(min=1)
        return loss.sum() / denom


# ===========================================================================
# 2) Paste this class into losses.py too
# ===========================================================================
class CombinedFocalLoss(nn.Module):
    """focal_weight * Focal + dice_weight * Dice, both ignoring void.

    Mirrors your CombinedLoss but swaps weighted-CE for weighted-Focal, so a
    focal run differs from the baseline ONLY in that one term.
    """

    def __init__(self, class_weights=None, gamma: float = 2.0,
                 focal_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = None):
        super().__init__()
        from dataset import VOID_INDEX
        idx = VOID_INDEX if ignore_index is None else ignore_index
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma,
                               ignore_index=idx)
        self.dice = DiceLoss(ignore_index=idx)  # your existing DiceLoss
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (self.focal_weight * self.focal(logits, target)
                + self.dice_weight * self.dice(logits, target))


# ===========================================================================
# 3) Add to train.py
# ===========================================================================
#
# (a) add a CLI flag near the other loss flags:
#
#     p.add_argument("--loss", default="ce_dice",
#                    choices=["ce_dice", "focal_dice"],
#                    help="loss type; focal_dice swaps CE for focal loss")
#     p.add_argument("--gamma", type=float, default=2.0,
#                    help="focal loss gamma (only used with --loss focal_dice)")
#
# (b) where you currently build the criterion, branch on the flag:
#
#     from losses import CombinedLoss, CombinedFocalLoss, load_class_weights
#     if args.loss == "focal_dice":
#         criterion = CombinedFocalLoss(
#             class_weights=class_weights, gamma=args.gamma,
#             focal_weight=args.ce_weight, dice_weight=args.dice_weight,
#         )
#     else:
#         criterion = CombinedLoss(
#             class_weights=class_weights,
#             ce_weight=args.ce_weight, dice_weight=args.dice_weight,
#         )
#
# Everything else (data, optimizer, schedule, AMP, metrics) stays identical,
# so a --loss focal_dice run is a clean one-variable change from baseline.
