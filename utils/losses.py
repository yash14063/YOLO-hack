from __future__ import annotations

"""
Segmentation losses that handle class imbalance.

- Focal Loss: down-weights easy pixels, focuses on hard / rare classes
- Weighted Cross Entropy: explicit per-class weights from inverse frequency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for semantic segmentation.

    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: float = 0.25,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("class_weight", weight if weight is not None else torch.ones(num_classes))
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: N,C,H,W  targets: N,H,W
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def compute_class_weights(mask_paths: list, num_classes: int, max_weight: float = 10.0) -> torch.Tensor:
    """
    Build inverse-frequency weights from a list of mask images (slow but clear for beginners).
    For speed, training code may use a running histogram instead.
    """
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    counts = np.zeros(num_classes, dtype=np.float64)
    for p in tqdm(mask_paths, desc="Class histogram"):
        m = np.array(Image.open(p))
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (num_classes * counts)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.5, max_weight)
    return torch.tensor(inv, dtype=torch.float32)


def build_loss(
    loss_type: str,
    num_classes: int,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    focal_alpha: float,
    device: torch.device,
    ignore_index: int = -100,
) -> nn.Module:
    weight = class_weights.to(device) if class_weights is not None else None
    if loss_type == "focal":
        return FocalLoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            alpha=focal_alpha,
            weight=weight,
            ignore_index=ignore_index,
        ).to(device)
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(device)
