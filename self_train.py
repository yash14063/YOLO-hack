"""
Self-training / pseudo-labeling loop:

1. Load the best SegFormer checkpoint.
2. Run inference on unlabeled images (e.g. new environments).
3. Keep pixels where the model is very confident; mark uncertain pixels as ignore (255).
4. Save pseudo masks and fine-tune a few epochs on labeled + pseudo data.

This is a classic semi-supervised technique that often improves generalization when
the unlabeled set resembles the target domain.

Usage:
  python self_train.py
  python self_train.py --epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    IMAGES_DIR,
    LOSS_TYPE,
    LOG_DIR,
    MASKS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS_SELF,
    PSEUDO_CONFIDENCE_THRESHOLD,
    PSEUDO_MIN_COVERAGE,
    PSEUDO_MASKS_DIR,
    STATIC_CHARTS_DIR,
    TRAIN_SIZES,
    UNLABELED_DIR,
)
from models.segformer_seg import SegFormerSegmenter, upsample_logits
from utils.augmentations import build_train_transform, build_val_transform
from utils.dataset import SegmentationDataset, UnlabeledImageDataset, list_image_mask_pairs
from utils.losses import build_loss, compute_class_weights
from utils.metrics import append_training_log, aggregate_iou_stats, fast_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("self_train")

IGNORE = 255


@torch.no_grad()
def generate_pseudo_masks(device: torch.device) -> int:
    """Write pseudo label PNGs into PSEUDO_MASKS_DIR. Returns count accepted."""
    ckpt = CHECKPOINT_DIR / "segformer_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError("Train SegFormer first: python train.py")

    PSEUDO_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    model = SegFormerSegmenter.load(ckpt, device)
    model.eval()

    if not UNLABELED_DIR.is_dir() or not any(UNLABELED_DIR.iterdir()):
        logger.warning(
            "No unlabeled images in %s — add JPG/PNG files for pseudo-labeling.",
            UNLABELED_DIR,
        )
        return 0

    from config import INFERENCE_SIZE

    ds = UnlabeledImageDataset(
        UNLABELED_DIR,
        transform=None,
    )
    if len(ds) == 0:
        return 0

    n_accepted = 0
    for i in tqdm(range(len(ds)), desc="Pseudo-labeling"):
        item = ds[i]
        stem = item["stem"]
        path = Path(item["path"])
        image = np.array(Image.open(path).convert("RGB"))
        h0, w0 = image.shape[:2]
        pil = Image.fromarray(image)
        pil = pil.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, H, W = inp.shape
        logits = model(inp)
        logits = upsample_logits(logits, (H, W))
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        pred = pred.cpu().numpy()[0]
        conf = conf.cpu().numpy()[0]
        coverage = float((conf >= PSEUDO_CONFIDENCE_THRESHOLD).mean())
        pseudo = pred.astype(np.uint8)
        pseudo[conf < PSEUDO_CONFIDENCE_THRESHOLD] = IGNORE

        if coverage < PSEUDO_MIN_COVERAGE:
            logger.debug("Skip %s (low coverage %.3f)", stem, coverage)
            continue

        out = Image.fromarray(pseudo, mode="L")
        out = out.resize((w0, h0), Image.NEAREST)
        out.save(PSEUDO_MASKS_DIR / f"{stem}.png")
        n_accepted += 1

    logger.info("Accepted %s pseudo-labeled images (see %s)", n_accepted, PSEUDO_MASKS_DIR)
    return n_accepted


def run_self_train(epochs: int, device: torch.device) -> dict:
    labeled_pairs = list_image_mask_pairs(IMAGES_DIR, MASKS_DIR)
    pseudo_pairs = list_image_mask_pairs(UNLABELED_DIR, PSEUDO_MASKS_DIR)
    all_pairs = labeled_pairs + pseudo_pairs
    if len(all_pairs) < 2:
        raise RuntimeError(
            "Need labeled data and at least one pseudo-labeled pair. "
            "Run train.py, add images to data/unlabeled/, then run self_train.py again."
        )

    random.shuffle(all_pairs)
    n_val = max(1, int(0.15 * len(all_pairs)))
    train_pairs = all_pairs[n_val:]
    val_pairs = all_pairs[:n_val]

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
        ignore_index=IGNORE,
    )

    ckpt = CHECKPOINT_DIR / "segformer_best.pt"
    model = SegFormerSegmenter.load(ckpt, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_iou = -1.0
    out_ckpt = CHECKPOINT_DIR / "segformer_selftrained.pt"

    for epoch in range(1, epochs + 1):
        side = random.choice(TRAIN_SIZES)
        train_tf = build_train_transform(side, side)
        train_ds = SegmentationDataset(train_pairs, transform=train_tf)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        model.train()
        total_loss = 0.0
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                logits = upsample_logits(logits, masks.shape[-2:])
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            hist += fast_confusion_matrix(preds.detach(), masks.detach(), NUM_CLASSES, ignore_index=IGNORE)

        n = len(train_loader.dataset)
        tr_loss = total_loss / max(n, 1)
        _, tr_iou = aggregate_iou_stats(hist)

        model.eval()
        va_loss = 0.0
        vhist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                logits = upsample_logits(logits, masks.shape[-2:])
                loss = criterion(logits, masks)
                va_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                vhist += fast_confusion_matrix(preds, masks, NUM_CLASSES, ignore_index=IGNORE)
        nv = len(val_loader.dataset)
        va_loss /= max(nv, 1)
        per_class, va_iou = aggregate_iou_stats(vhist)

        row = {
            "phase": "self_train",
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_mIoU": tr_iou,
            "val_mIoU": va_iou,
            "per_class_iou": {CLASS_NAMES[i]: float(per_class[i]) for i in range(NUM_CLASSES)},
        }
        history.append(row)
        append_training_log(LOG_DIR, row)
        logger.info(
            "Self-train %s/%s | tr loss %.4f | val loss %.4f | val mIoU %.4f",
            epoch,
            epochs,
            tr_loss,
            va_loss,
            va_iou,
        )
        if va_iou > best_iou:
            best_iou = va_iou
            model.save(out_ckpt)

    # Promote best self-trained weights for /predict (keep one-time baseline backup)
    best_pt = CHECKPOINT_DIR / "segformer_best.pt"
    baseline_pt = CHECKPOINT_DIR / "segformer_baseline.pt"
    if out_ckpt.is_file():
        if best_pt.is_file() and not baseline_pt.is_file():
            shutil.copy2(best_pt, baseline_pt)
        shutil.copy2(out_ckpt, best_pt)

    after = {
        "best_val_mIoU_after_self_train": best_iou,
        "checkpoint": str(out_ckpt),
        "per_class_last": history[-1]["per_class_iou"] if history else {},
    }
    with open(LOG_DIR / "metrics_after_self_train.json", "w", encoding="utf-8") as f:
        json.dump(after, f, indent=2)

    # Simple before/after: read previous summary if present
    summary_seg = LOG_DIR / "summary_segformer.json"
    before_m = None
    if summary_seg.is_file():
        with open(summary_seg, encoding="utf-8") as f:
            before_m = json.load(f).get("best_val_mIoU")
        before_json = LOG_DIR / "metrics_before_self_train.json"
        if not before_json.is_file():
            shutil.copy2(summary_seg, before_json)
    with open(LOG_DIR / "before_after_self_train.json", "w", encoding="utf-8") as f:
        json.dump({"before_mIoU": before_m, "after_mIoU": best_iou}, f, indent=2)

    return after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_SELF)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    n = generate_pseudo_masks(device)
    if n == 0 and not any(PSEUDO_MASKS_DIR.glob("*.png")):
        logger.warning(
            "No pseudo masks created — self-training will only use labeled data "
            "(still runs fine for demo)."
        )
    result = run_self_train(args.epochs, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
