"""
DeepLabV3+ (ResNet-50 backbone) from torchvision — optional second model for ensembling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

logger = logging.getLogger(__name__)


class DeepLabSegmenter(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        weights = None
        if pretrained:
            try:
                from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

                weights = DeepLabV3_ResNet50_Weights.DEFAULT
            except Exception:
                weights = True  # older torchvision
        self.net = deeplabv3_resnet50(weights=weights, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: N,3,H,W in 0..1 — DeepLab expects ImageNet normalized tensors internally.
        We apply normalization here for a single consistent API with SegFormer path.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        xn = (x - mean) / std
        out = self.net(xn)["out"]
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "num_classes": self.num_classes}, path)
        logger.info("Saved DeepLab checkpoint to %s", path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> DeepLabSegmenter:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        m = cls(num_classes=ckpt["num_classes"], pretrained=False)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device)
        return m
