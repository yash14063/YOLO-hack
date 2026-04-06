"""
Domain randomization augmentations for synthetic → real generalization.

We simulate lighting changes, sensor noise, blur, color shifts, and dust/fog
so the model learns features that are stable under environment change.
"""
import random

import albumentations as A
import cv2
import numpy as np

from config import AUGMENT_STRENGTH, NUM_CLASSES


def _strength(x: float) -> float:
    """Scale augmentation probability/intensity by global strength."""
    return max(0.0, min(1.0, x * AUGMENT_STRENGTH))


def build_train_transform(image_height: int, image_width: int) -> A.Compose:
    """
    Strong augmentations for training (domain randomization).

    - Brightness/contrast: simulates sun angle and exposure
    - Gaussian noise: sensor / compression noise
    - Blur: motion or cheap optics
    - Hue/saturation/value: different cameras / atmospheres
    - Fog / dust: CLAHE + RandomFog (if available) approximates hazy desert air
    """
    s = _strength

    steps = [
        A.RandomResizedCrop(
            size=(image_height, image_width),
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.35 * s(1.0),
            contrast_limit=0.35 * s(1.0),
            p=s(0.85),
        ),
        A.GaussNoise(p=s(0.5)),
        A.GaussianBlur(blur_limit=(3, 7), p=s(0.35)),
        A.HueSaturationValue(
            hue_shift_limit=int(20 * s(1.0)),
            sat_shift_limit=int(40 * s(1.0)),
            val_shift_limit=int(30 * s(1.0)),
            p=s(0.75),
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=s(0.4)),
    ]
    # RandomFog exists in recent albumentations; fall back to extra brightness wash
    if hasattr(A, "RandomFog"):
        steps.append(
            A.RandomFog(
                p=s(0.45),
            )
        )
    else:
        steps.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.25 * s(1.0),
                contrast_limit=0.1 * s(1.0),
                p=s(0.35),
            )
        )

    return A.Compose(steps, additional_targets={"mask": "mask"})


def build_val_transform(image_height: int, image_width: int) -> A.Compose:
    """Resize only — no randomness for validation / stable metrics."""
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
        ],
        additional_targets={"mask": "mask"},
    )


def apply_fog_dust_numpy(image: np.ndarray) -> np.ndarray:
    """
    Optional extra 'dust' pass (used rarely); keeps code readable for demos.
    image: HWC uint8 RGB
    """
    if random.random() > 0.3 * _strength(1.0):
        return image
    h, w = image.shape[:2]
    overlay = np.random.randint(180, 255, (h, w, 3), dtype=np.uint8)
    dust = cv2.GaussianBlur(overlay, (0, 0), sigmaX=w * 0.05)
    alpha = 0.08 + random.random() * 0.12
    out = (image.astype(np.float32) * (1 - alpha) + dust.astype(np.float32) * alpha).clip(
        0, 255
    ).astype(np.uint8)
    return out


def mask_to_multichannel(mask: np.ndarray) -> np.ndarray:
    """One-hot mask for albumentations (not always needed); kept for extensions."""
    # mask: H,W with values 0..NUM_CLASSES-1
    oh = np.zeros((*mask.shape, NUM_CLASSES), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        oh[:, :, c] = (mask == c).astype(np.uint8) * 255
    return oh
