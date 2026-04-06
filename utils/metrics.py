"""
IoU (Intersection over Union) metrics for semantic segmentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def compute_iou_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> tuple[np.ndarray, float]:
    """
    pred, target: (N, H, W) long tensors
    Returns per-class IoU and mean IoU (excluding ignore and absent classes in GT).
    """
    pred = pred.flatten()
    target = target.flatten()
    if ignore_index >= 0:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]

    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)
    arr = np.array(ious, dtype=np.float64)
    mean_iou = float(np.nanmean(arr))
    return arr, mean_iou


def aggregate_iou_stats(
    hist: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    hist[c_true, c_pred] = count
    """
    num_classes = hist.shape[0]
    ious = []
    for c in range(num_classes):
        tp = hist[c, c]
        fp = hist[:, c].sum() - tp
        fn = hist[c, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            ious.append(float("nan"))
        else:
            ious.append(tp / denom)
    arr = np.array(ious, dtype=np.float64)
    return arr, float(np.nanmean(arr))


def fast_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """pred, target: N,H,W"""
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    mask = (target >= 0) & (target < num_classes) & (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    hist = np.bincount(
        num_classes * target.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def append_training_log(log_dir: Path, entry: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "training_history.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_training_history(log_dir: Path) -> list[dict]:
    path = log_dir / "training_history.jsonl"
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
