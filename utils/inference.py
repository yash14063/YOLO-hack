"""
Run segmentation forward pass + optional softmax confidence maps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CHECKPOINT_DIR, INFERENCE_SIZE, NUM_CLASSES
from models.segformer_seg import SegFormerSegmenter, upsample_logits

logger = logging.getLogger(__name__)


def load_model_for_inference(
    device: torch.device,
    segformer_path: Path | None = None,
    deeplab_path: Path | None = None,
    use_ensemble: bool = False,
):
    """
    Load SegFormer (required). Optionally load DeepLab + ensemble.
    """
    sf_path = segformer_path or (CHECKPOINT_DIR / "segformer_best.pt")
    if not sf_path.is_file():
        raise FileNotFoundError(
            f"No SegFormer weights at {sf_path}. Train first (python train.py)."
        )
    segformer = SegFormerSegmenter.load(sf_path, device)
    if use_ensemble and deeplab_path and Path(deeplab_path).is_file():
        from models.deeplab_seg import DeepLabSegmenter
        from models.ensemble import EnsembleSegmenter

        deeplab = DeepLabSegmenter.load(Path(deeplab_path), device)
        return EnsembleSegmenter(segformer, deeplab).eval()
    return segformer.eval()


def preprocess_pil(image: Image.Image, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize RGB PIL -> tensor N,3,H,W in [0,1]. Returns (tensor, original_hw)."""
    orig = image.size[::-1]  # H,W
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t, (orig[0], orig[1])


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    size: int | None = None,
    return_prob: bool = False,
) -> tuple[np.ndarray, float | None]:
    """
    Returns (mask HxW int64 class ids at original resolution, mean_confidence or None).
    """
    size = size or INFERENCE_SIZE
    inp, orig_hw = preprocess_pil(image, size)
    inp = inp.to(device)
    _, _, H, W = inp.shape

    from models.ensemble import EnsembleSegmenter

    if isinstance(model, EnsembleSegmenter):
        logits = model(inp)
    elif isinstance(model, SegFormerSegmenter):
        logits = model(inp)
        logits = upsample_logits(logits, (H, W))
    else:
        logits = model(inp)
        if logits.shape[-2:] != (H, W):
            logits = upsample_logits(logits, (H, W))

    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    pred = pred.cpu().numpy()[0]
    conf_map = conf.cpu().numpy()[0]
    mean_conf = float(conf_map.mean())

    # Resize pred to original image size
    pred_pil = Image.fromarray(pred.astype(np.uint8))
    pred_pil = pred_pil.resize((image.size[0], image.size[1]), Image.NEAREST)
    mask_out = np.array(pred_pil).astype(np.int64)

    if return_prob:
        return mask_out, mean_conf
    return mask_out, mean_conf


def colorize_mask(mask: np.ndarray, palette: list[tuple[int, int, int]] | None = None) -> Image.Image:
    """Map class indices to RGB for visualization."""
    if palette is None:
        palette = [
            (0, 0, 0),
            (34, 139, 34),
            (139, 69, 19),
            (128, 128, 128),
            (255, 105, 180),
        ]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(palette[:NUM_CLASSES]):
        rgb[mask == c] = col
    return Image.fromarray(rgb)
