"""
Train SegFormer (+ optional DeepLab) with:
- Domain randomization augmentations
- Focal loss or weighted CE
- Multi-scale crops (random choice from config.TRAIN_SIZES)

Usage (from project root):
  python train.py
  python train.py --epochs 12 --model both
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LOSS_TYPE,
    LOG_DIR,
    NUM_CLASSES,
    NUM_EPOCHS_BASE,
    SEGFORMER_ID,
    STATIC_CHARTS_DIR,
    TRAIN_SIZES,
    USE_ENSEMBLE,
)
from models.deeplab_seg import DeepLabSegmenter
from models.segformer_seg import SegFormerSegmenter, upsample_logits
from utils.augmentations import build_train_transform, build_val_transform
from utils.dataset import SegmentationDataset, list_image_mask_pairs
from utils.losses import build_loss, compute_class_weights
from utils.metrics import append_training_log, fast_confusion_matrix, aggregate_iou_stats

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("train")
_fh = logging.FileHandler(LOG_DIR / "train.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_fh)


def pick_random_train_size() -> int:
    return random.choice(TRAIN_SIZES)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, is_segformer: bool):
    model.train()
    total_loss = 0.0
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if is_segformer:
                logits = model(images)
                logits = upsample_logits(logits, masks.shape[-2:])
            else:
                logits = model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        hist += fast_confusion_matrix(
            preds.detach(), masks.detach(), NUM_CLASSES, ignore_index=255
        )

    n = len(loader.dataset)
    _, m_iou = aggregate_iou_stats(hist)
    return total_loss / max(n, 1), m_iou


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_segformer: bool):
    model.eval()
    total_loss = 0.0
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if is_segformer:
            logits = model(images)
            logits = upsample_logits(logits, masks.shape[-2:])
        else:
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        hist += fast_confusion_matrix(preds, masks, NUM_CLASSES, ignore_index=255)
    n = len(loader.dataset)
    per_class, m_iou = aggregate_iou_stats(hist)
    return total_loss / max(n, 1), m_iou, per_class


def plot_curves(history: list[dict], out_path: Path, title: str) -> None:
    STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_iou = [h["val_mIoU"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, label="train loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="val loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_iou, label="val mIoU", color="tab:green")
    ax2.set_ylabel("mIoU")
    ax2.legend(loc="lower right")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved chart: %s", out_path)


def run_training(
    images_dir: Path,
    masks_dir: Path,
    epochs: int,
    model_name: str,
    device: torch.device,
) -> dict:
    pairs = list_image_mask_pairs(images_dir, masks_dir)
    if len(pairs) < 2:
        raise RuntimeError(
            f"Need at least 2 image/mask pairs in {images_dir} / {masks_dir}. "
            "Run: python scripts/create_sample_data.py"
        )

    random.shuffle(pairs)
    n_val = max(1, int(0.15 * len(pairs)))
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    # Class weights from training masks (quick histogram)
    mask_paths = [p[1] for p in train_pairs]
    class_weights = compute_class_weights(mask_paths, NUM_CLASSES)

    size_val = max(TRAIN_SIZES)
    val_tf = build_val_transform(size_val, size_val)
    val_ds = SegmentationDataset(val_pairs, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = build_loss(
        LOSS_TYPE,
        NUM_CLASSES,
        class_weights,
        FOCAL_GAMMA,
        FOCAL_ALPHA,
        device,
        ignore_index=255,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_iou = -1.0
    best_path = CHECKPOINT_DIR / (
        "segformer_best.pt" if model_name == "segformer" else "deeplab_best.pt"
    )

    is_segformer = model_name == "segformer"
    if is_segformer:
        model = SegFormerSegmenter(NUM_CLASSES, SEGFORMER_ID, cache_dir=CHECKPOINT_DIR / "hf_cache")
    else:
        model = DeepLabSegmenter(NUM_CLASSES, pretrained=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        side = pick_random_train_size()
        train_tf = build_train_transform(side, side)
        train_ds = SegmentationDataset(train_pairs, transform=train_tf)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
        )

        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, is_segformer
        )
        va_loss, va_iou, per_class = evaluate(
            model, val_loader, criterion, device, is_segformer
        )

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_mIoU": tr_iou,
            "val_mIoU": va_iou,
            "model": model_name,
            "per_class_iou": {CLASS_NAMES[i]: float(per_class[i]) for i in range(NUM_CLASSES)},
        }
        history.append(row)
        append_training_log(LOG_DIR, row)
        logger.info(
            "Epoch %s/%s | train loss %.4f | val loss %.4f | val mIoU %.4f",
            epoch,
            epochs,
            tr_loss,
            va_loss,
            va_iou,
        )

        if va_iou > best_iou:
            best_iou = va_iou
            model.save(best_path)

    chart_name = f"training_{model_name}.png"
    plot_curves(history, STATIC_CHARTS_DIR / chart_name, f"Training — {model_name}")

    summary = {
        "best_val_mIoU": best_iou,
        "checkpoint": str(best_path),
        "history_file": str(LOG_DIR / "training_history.jsonl"),
        "chart": str(STATIC_CHARTS_DIR / chart_name),
    }
    with open(LOG_DIR / f"summary_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_BASE)
    parser.add_argument(
        "--model",
        choices=["segformer", "deeplab", "both"],
        default="segformer",
    )
    args = parser.parse_args()

    from config import IMAGES_DIR, MASKS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    results = {}
    if args.model in ("segformer", "both"):
        results["segformer"] = run_training(
            IMAGES_DIR, MASKS_DIR, args.epochs, "segformer", device
        )
    if args.model in ("deeplab", "both") or USE_ENSEMBLE:
        results["deeplab"] = run_training(
            IMAGES_DIR, MASKS_DIR, max(4, args.epochs // 2), "deeplab", device
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
