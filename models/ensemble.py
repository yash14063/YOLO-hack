"""
Average logits from SegFormer and DeepLab for optional ensemble inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.deeplab_seg import DeepLabSegmenter
from models.segformer_seg import SegFormerSegmenter, upsample_logits


class EnsembleSegmenter(nn.Module):
    def __init__(self, segformer: SegFormerSegmenter, deeplab: DeepLabSegmenter):
        super().__init__()
        self.segformer = segformer
        self.deeplab = deeplab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N,3,H,W 0..1
        _, _, H, W = x.shape
        lf = self.segformer(x)
        lf = upsample_logits(lf, (H, W))
        ld = self.deeplab(x)
        if ld.shape[-2:] != (H, W):
            ld = upsample_logits(ld, (H, W))
        return 0.5 * (lf + ld)
