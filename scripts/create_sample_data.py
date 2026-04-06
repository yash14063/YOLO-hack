"""
Generate tiny synthetic "desert" RGB images + PNG masks for a zero-setup demo.

Classes (mask pixel values):
  0 background (sand)
  1 trees   (green blobs)
  2 logs    (brown rectangles)
  3 rocks   (gray blobs)
  4 flowers (pink dots)

Run from project root:
  python scripts/create_sample_data.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "data" / "images"
MASKS = ROOT / "data" / "masks"
UNLABELED = ROOT / "data" / "unlabeled"
TEST = ROOT / "sample_test_images"


def synth_one(idx: int, size: tuple[int, int] = (384, 384), seed: int | None = None) -> tuple[Image.Image, Image.Image]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    w, h = size
    # Sand background with mild gradient (domain: synthetic desert)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        base[y, :] = [210 - y // 8, 180 - y // 12, 120 - y // 16]
    noise = np.random.randint(-25, 25, base.shape, dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)

    # Trees (green circles)
    for _ in range(random.randint(2, 5)):
        cx, cy = random.randint(40, w - 40), random.randint(40, h - 40)
        r = random.randint(18, 40)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(34, 100, 34))
        md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)

    # Logs (brown rects)
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(10, w - 80), random.randint(10, h - 40)
        x2, y2 = x1 + random.randint(50, 120), y1 + random.randint(15, 35)
        draw.rectangle([x1, y1, x2, y2], fill=(101, 67, 33))
        md.rectangle([x1, y1, x2, y2], fill=2)

    # Rocks (gray blobs)
    for _ in range(random.randint(2, 6)):
        cx, cy = random.randint(30, w - 30), random.randint(30, h - 30)
        r = random.randint(8, 22)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 120, 120))
        md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=3)

    # Flowers (small pink)
    for _ in range(random.randint(3, 10)):
        cx, cy = random.randint(20, w - 20), random.randint(20, h - 20)
        r = random.randint(4, 9)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 105, 180))
        md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=4)

    return img, mask


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    UNLABELED.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)

    n_train = 24
    for i in range(n_train):
        img, m = synth_one(i, seed=1000 + i)
        stem = f"desert_{i:03d}"
        img.save(IMAGES / f"{stem}.png")
        m.save(MASKS / f"{stem}.png")

    # A few "unlabeled" frames (no masks) for the self-training story
    for j in range(6):
        img, _ = synth_one(100 + j, seed=9000 + j, size=(320, 320))
        img.save(UNLABELED / f"unlabeled_{j:03d}.png")

    # Explicit test folder for README / judges
    for k in range(3):
        img, _ = synth_one(200 + k, seed=7000 + k)
        img.save(TEST / f"test_{k:03d}.png")

    print(f"Wrote {n_train} labeled pairs to {IMAGES} and {MASKS}")
    print(f"Wrote unlabeled images to {UNLABELED}")
    print(f"Wrote sample test images to {TEST}")


if __name__ == "__main__":
    main()
