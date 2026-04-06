"""
Flask REST endpoints for training, self-training, and segmentation inference.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    OUTPUT_MASKS_DIR,
    ROOT,
    STATIC_CHARTS_DIR,
    USE_ENSEMBLE,
)
from utils.inference import colorize_mask, load_model_for_inference, predict_image
from utils.metrics import fast_confusion_matrix, aggregate_iou_stats
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            deeplab_path = CHECKPOINT_DIR / "deeplab_best.pt"
            _model = load_model_for_inference(
                _device,
                use_ensemble=USE_ENSEMBLE and deeplab_path.is_file(),
                deeplab_path=deeplab_path,
            )
        return _model


def _run_script_async(args: list[str]) -> None:
    def job():
        try:
            subprocess.run(
                [sys.executable, *args],
                cwd=str(ROOT),
                check=False,
            )
        except Exception as e:
            logger.exception("Background job failed: %s", e)

    threading.Thread(target=job, daemon=True).start()


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(_device)})


@bp.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "message": "Segmentation API is running",
            "docs": {
                "health": "/health",
                "predict": "POST /predict (multipart: image, optional ground_truth)",
                "train": "POST /train",
                "self_train": "POST /self-train",
                "metrics": "/metrics",
            },
            "frontend_hint": "Open http://localhost:5173 for dashboard UI",
        }
    )


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Multipart form: field `image` (file).
    Returns JSON: colored mask as base64 PNG, optional mean confidence, saved paths.
    Optional form field `ground_truth`: mask PNG for IoU (same size as image).
    """
    if "image" not in request.files:
        return jsonify({"error": "missing file field `image`"}), 400
    f = request.files["image"]
    try:
        image = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    mask, mean_conf = predict_image(model, image, _device, return_prob=True)
    colored = colorize_mask(mask)

    OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(f.filename or "upload").stem or "upload"
    mask_path = OUTPUT_MASKS_DIR / f"{stem}_mask.png"
    overlay_path = OUTPUT_MASKS_DIR / f"{stem}_overlay.png"
    colored.save(mask_path)

    blend = Image.blend(image, colored, 0.45)
    blend.save(overlay_path)

    buf = io.BytesIO()
    colored.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out = {
        "mask_png_base64": b64,
        "mean_confidence": mean_conf,
        "saved_mask_path": str(mask_path),
        "saved_overlay_path": str(overlay_path),
    }

    if "ground_truth" in request.files and request.files["ground_truth"].filename:
        gt_file = request.files["ground_truth"]
        gt = np.array(Image.open(gt_file.stream).convert("L"))
        if gt.shape != mask.shape:
            gt_im = Image.fromarray(gt.astype(np.uint8), mode="L")
            gt_im = gt_im.resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
            gt = np.array(gt_im)
        from config import NUM_CLASSES

        pred_t = torch.from_numpy(mask).unsqueeze(0)
        gt_t = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
        hist = fast_confusion_matrix(pred_t, gt_t, NUM_CLASSES, ignore_index=255)
        _, miou = aggregate_iou_stats(hist)
        out["iou_mean"] = miou

    return jsonify(out)


@bp.route("/train", methods=["POST"])
def train():
    """
    Starts baseline training in a background thread (non-blocking for the UI).
    Optional JSON body: {"epochs": 8, "model": "segformer"|"deeplab"|"both"}
    """
    body = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 8))
    model = body.get("model", "segformer")
    cmd = ["train.py", "--epochs", str(epochs), "--model", str(model)]
    _run_script_async(cmd)
    return jsonify({"status": "started", "command": " ".join(cmd)})


@bp.route("/self-train", methods=["POST"])
@bp.route("/self_train", methods=["POST"])
def self_train():
    body = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 5))
    _run_script_async(["self_train.py", "--epochs", str(epochs)])
    return jsonify({"status": "started", "message": "Pseudo-label + fine-tune running in background"})


@bp.route("/metrics", methods=["GET"])
def metrics():
    """Training history (JSONL parsed) + chart URLs if present."""
    hist_path = LOG_DIR / "training_history.jsonl"
    rows = []
    if hist_path.is_file():
        with open(hist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    charts = sorted(STATIC_CHARTS_DIR.glob("*.png")) if STATIC_CHARTS_DIR.is_dir() else []
    summary_seg = LOG_DIR / "summary_segformer.json"
    summary = {}
    if summary_seg.is_file():
        with open(summary_seg, encoding="utf-8") as f:
            summary["segformer"] = json.load(f)
    return jsonify(
        {
            "history": rows[-200:],
            "charts": [str(p) for p in charts],
            "summary": summary,
        }
    )


@bp.route("/before-after", methods=["GET"])
def before_after():
    p = LOG_DIR / "before_after_self_train.json"
    if not p.is_file():
        return jsonify({"before_mIoU": None, "after_mIoU": None, "note": "Run self-training once to populate"})
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@bp.route("/charts/<path:name>", methods=["GET"])
def chart_file(name):
    safe = Path(name).name
    path = STATIC_CHARTS_DIR / safe
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@bp.route("/reload-model", methods=["POST"])
def reload_model():
    """Call after training completes to load new weights without restarting Flask."""
    global _model
    with _model_lock:
        _model = None
    get_model()
    return jsonify({"status": "model reloaded"})
