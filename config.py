"""
Central configuration for the segmentation hackathon project.
Adjust paths and hyperparameters here before training.
"""
from pathlib import Path

# Project root (this file lives at project root)
ROOT = Path(__file__).resolve().parent

# --- Data layout (create folders or run scripts/create_sample_data.py) ---
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
UNLABELED_DIR = DATA_DIR / "unlabeled"
PSEUDO_MASKS_DIR = DATA_DIR / "pseudo_masks"
OUTPUT_MASKS_DIR = ROOT / "outputs" / "predicted_masks"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
STATIC_CHARTS_DIR = ROOT / "static" / "charts"

# Class names: index 0 = background (standard for segmentation)
CLASS_NAMES = ["background", "trees", "logs", "rocks", "flowers"]
NUM_CLASSES = len(CLASS_NAMES)

# Training
BATCH_SIZE = 4
NUM_EPOCHS_BASE = 8
NUM_EPOCHS_SELF = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
# Multi-scale training: model sees crops at these sizes (longest side)
TRAIN_SIZES = [256, 512]
INFERENCE_SIZE = 512

# Domain randomization / augmentation strength (0 = off, 1 = strong)
AUGMENT_STRENGTH = 1.0

# Self-training: only pixels with max probability above this become pseudo-labels
PSEUDO_CONFIDENCE_THRESHOLD = 0.92
# Minimum fraction of high-confidence pixels in an image to accept it
PSEUDO_MIN_COVERAGE = 0.15

# Loss: "focal" or "weighted_ce"
LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Models
SEGFORMER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
USE_ENSEMBLE = False  # Set True to train & average with DeepLabV3+

# Device (overridden by train script if CUDA available)
DEVICE = "cuda"

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
MAX_CONTENT_LENGTH_MB = 32
