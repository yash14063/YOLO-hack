"""
SegFormer (Transformers) wrapper for semantic segmentation.

We start from an NVIDIA ADE20k-pretrained checkpoint and replace the classification
head so the channel count matches our desert classes (trees, logs, rocks, flowers).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

logger = logging.getLogger(__name__)


class SegFormerSegmenter(nn.Module):
    def __init__(self, num_classes: int, model_id: str, cache_dir: Path | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.model_id = model_id
        kwargs = {"num_labels": num_classes, "ignore_mismatched_sizes": True}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.net = SegformerForSemanticSegmentation.from_pretrained(model_id, **kwargs)
        proc_kw = {}
        if cache_dir is not None:
            proc_kw["cache_dir"] = str(cache_dir)
        self.processor = SegformerImageProcessor.from_pretrained(model_id, **proc_kw)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: N,3,H,W in 0..1 float (we normalize inside using processor stats).
        Returns logits N,C,h,w (smaller than input — upsample in training/inference).
        """
        # HF expects normalized inputs; manual normalize like ImageNet stats used in SegFormer
        mean = torch.tensor(self.processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)
        x = (pixel_values - mean) / std
        out = self.net(pixel_values=x)
        logits = out.logits
        return logits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "num_classes": self.num_classes, "model_id": self.model_id}, path)
        logger.info("Saved SegFormer checkpoint to %s", path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> SegFormerSegmenter:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        m = cls(num_classes=ckpt["num_classes"], model_id=ckpt["model_id"])
        m.load_state_dict(ckpt["state_dict"])
        m.to(device)
        return m


def upsample_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """logits: N,C,h,w -> N,C,H,W"""
    return torch.nn.functional.interpolate(
        logits, size=target_size, mode="bilinear", align_corners=False
    )
