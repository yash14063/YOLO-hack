"""
Flask entrypoint — Domain-Adaptive Self-Improving Image Segmentation API.

Run from project root:
  python app.py

Endpoints (see routes/api.py):
  GET  /health
  POST /predict
  POST /train
  POST /self-train
  GET  /metrics
  GET  /before-after
  POST /reload-model
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, MAX_CONTENT_LENGTH_MB, ROOT
from routes.api import bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(MAX_CONTENT_LENGTH_MB) * 1024 * 1024
    CORS(app)
    app.register_blueprint(bp)
    return app


app = create_app()


if __name__ == "__main__":
    # Ensure artifact folders exist for a smooth first run
    for sub in ("data/images", "data/masks", "data/unlabeled", "outputs/predicted_masks", "logs", "static/charts", "checkpoints"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
