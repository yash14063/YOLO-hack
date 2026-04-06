"""
PyTorch Dataset for image + mask pairs (PNG masks with class indices 0..C-1).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def list_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files to masks by stem (e.g. img_01.png ↔ img_01.png)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.is_file():
            mask_path = masks_dir / f"{img_path.stem}.jpg"
        if mask_path.is_file():
            pairs.append((img_path, mask_path))
        else:
            logger.warning("No mask for image: %s", img_path.name)
    return pairs


class SegmentationDataset(Dataset):
    """
    Loads RGB images and single-channel masks.
    `transform` should be albumentations.Compose with additional_targets mask.
    """

    def __init__(self, pairs: list[tuple[Path, Path]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Albumentations may return mask as float; cast back
        mask = np.ascontiguousarray(mask).astype(np.int64)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return {"image": image_t, "mask": mask_t, "stem": img_path.stem}


class UnlabeledImageDataset(Dataset):
    """Images only — for pseudo-labeling."""

    def __init__(self, folder: Path, transform=None):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        image = np.array(Image.open(p).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"image": image_t, "path": str(p), "stem": p.stem}
