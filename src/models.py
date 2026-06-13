"""Model factory: one interface for both architectures.

build_model("unet")      -> U-Net, ResNet34 encoder, ImageNet weights (smp)
build_model("segformer") -> SegFormer MiT-B2, ImageNet weights (HF)

Both wrappers return logits at the FULL input resolution (N, C, H, W), so
the training loop, loss and metrics are identical for the two models -
this is what keeps the comparison fair. SegFormer natively outputs logits
at 1/4 resolution, so its wrapper upsamples them with bilinear
interpolation, which is the standard SegFormer training recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES


class SegformerWrapper(nn.Module):
    """Wrap HF SegformerForSemanticSegmentation to return full-res logits."""

    def __init__(self, num_classes: int, checkpoint: str = "nvidia/mit-b2"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # replace the pretrained head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF returns logits at H/4 x W/4; upsample to the input resolution.
        logits = self.net(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    name = name.lower()
    if name == "unet":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if name == "segformer":
        return SegformerWrapper(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Use 'unet' or 'segformer'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
