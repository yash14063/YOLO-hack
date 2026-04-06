# Domain-Adaptive Self-Improving Image Segmentation System

Hackathon project: segment **trees, logs, rocks, and flowers** in desert-style imagery, with **domain randomization**, **focal loss**, **multi-scale training**, optional **DeepLabV3+ ensemble**, a **Flask API**, and a **React dashboard**.

## Why this approach wins (short pitch)

1. **Domain randomization** — The model never sees only one “camera” or weather. Heavy photometric augmentations (brightness, noise, blur, color shift, haze/fog) force invariance to appearance changes, which helps **synthetic → real** and **seen → unseen** environments.
2. **Self-training (pseudo-labeling)** — After supervised training, the model labels **unlabeled** target-domain images where it is confident. Those pixels become extra training signal, nudging the decision boundary toward the deployment domain without new manual masks.
3. **Imbalanced classes** — **Focal loss** (or weighted CE) prevents huge background or sand regions from drowning rare classes (e.g. flowers).
4. **Multi-scale** — Random **256 / 512** training sizes improve scale robustness (small rocks vs large trees).

---

## Domain adaptation (plain language)

**Domain adaptation** means reducing the gap between **training data** (e.g. synthetic desert) and **test / deployment data** (new biome, lighting, or sensor). We do it two ways:

- **Simulation-side:** domain randomization makes the training distribution *broad* so the test domain is more likely to fall inside what the model has already seen in variation.
- **Target-side:** self-training uses **unlabeled** images from the target environment. High-confidence predictions act as **pseudo-labels**, so the model adapts to statistics of the new domain without needing full new annotations.

---

## Pseudo-labeling (plain language)

1. Train (or load) a teacher model on labeled pairs `(image, mask)`.
2. Run the teacher on **unlabeled** images.
3. For each pixel, look at the predicted probability. If it is **above a threshold**, we trust the class; otherwise we mark the pixel as **ignore** (not used in the loss).
4. **Fine-tune** on **original labels + pseudo-labeled** images.

Noisy pseudo-labels can hurt training, so we use **high thresholds** and **minimum coverage** filters (see `config.py`).

---

## Project layout

```
├── app.py                 # Flask server
├── train.py               # Baseline (+ optional DeepLab) training
├── self_train.py          # Pseudo-label generation + fine-tune
├── config.py              # Paths, classes, hyperparameters
├── requirements.txt
├── models/
│   ├── segformer_seg.py   # SegFormer (Hugging Face)
│   ├── deeplab_seg.py     # DeepLabV3+ (torchvision)
│   └── ensemble.py        # Average logits (bonus)
├── utils/
│   ├── dataset.py
│   ├── augmentations.py   # Domain randomization
│   ├── losses.py          # Focal / weighted CE
│   ├── metrics.py         # IoU, logging
│   └── inference.py
├── routes/
│   └── api.py             # REST endpoints
├── scripts/
│   └── create_sample_data.py
├── data/
│   ├── images/            # Labeled RGB
│   ├── masks/             # PNG, pixel values = class id (0..4)
│   └── unlabeled/         # RGB only (for self-training)
├── sample_test_images/    # Populated by the sample script
├── checkpoints/           # Saved .pt weights (created on train)
├── logs/                  # training_history.jsonl, charts data
├── static/charts/         # Matplotlib PNG curves
└── frontend/              # React (Vite) dashboard
```

### Class IDs (masks)

| Value | Class       |
|------:|-------------|
| 0     | background  |
| 1     | trees       |
| 2     | logs        |
| 3     | rocks       |
| 4     | flowers     |

For pseudo-labels, **255** = ignore in loss.

---

## Step-by-step: run locally

### 1. Python environment

```bash
cd "c:\Users\HP\Desktop\Hackethon_Team\YOLO hack"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `torch` fails on your machine, install the wheel from [pytorch.org](https://pytorch.org) for your CUDA/CPU setup, then `pip install -r requirements.txt` again (pip may skip torch if already satisfied).

### 2. Sample dataset + test images

```bash
python scripts/create_sample_data.py
```

This fills `data/images`, `data/masks`, `data/unlabeled`, and `sample_test_images/`.

### 3. Train SegFormer

```bash
python train.py --epochs 6 --model segformer
```

First run **downloads** the Hugging Face SegFormer weights (needs internet).

Outputs:

- `checkpoints/segformer_best.pt`
- `logs/training_history.jsonl`, `logs/summary_segformer.json`
- `static/charts/training_segformer.png`

### 4. (Optional) Train DeepLabV3+ for ensemble

```bash
python train.py --epochs 6 --model deeplab
```

Set `USE_ENSEMBLE = True` in `config.py` after both checkpoints exist. The API will average logits.

### 5. Self-training loop

```bash
python self_train.py --epochs 4
```

Writes pseudo masks to `data/pseudo_masks/`, fine-tunes, saves `checkpoints/segformer_selftrained.pt`, updates `segformer_best.pt`, and writes `logs/before_after_self_train.json`.

### 6. Flask API

```bash
python app.py
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + device |
| POST | `/predict` | Form field `image` (file); optional `ground_truth` mask for IoU |
| POST | `/train` | JSON `{"epochs": 8, "model": "segformer"}` — runs in background |
| POST | `/self-train` | JSON `{"epochs": 5}` — pseudo-label + fine-tune in background |
| GET | `/metrics` | Parsed training history + chart paths |
| GET | `/before-after` | mIoU before vs after self-training |
| POST | `/reload-model` | Load latest weights without restart |

### 7. React dashboard

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (e.g. `http://localhost:5173`). The Vite dev server **proxies** API calls to `http://127.0.0.1:5000`.

---

## Using your own dataset

1. Put RGB images in `data/images/`.
2. Put matching single-channel PNG masks in `data/masks/` with the **same filename stem** (e.g. `scene42.jpg` ↔ `scene42.png`).
3. Put target-domain **unlabeled** images in `data/unlabeled/` for self-training.
4. If you change the number of classes, update `CLASS_NAMES` and `NUM_CLASSES` in `config.py` and retrain.

---

## Team storyboard (demo flow)

1. Show **domain randomization** list in `utils/augmentations.py`.
2. Train (or show pre-trained) — **mIoU** in logs / React chart.
3. Upload **sample_test_images/test_000.png** — show overlay.
4. Click **Improve model** — explain pseudo-labels on `data/unlabeled/`.
5. **Reload weights** — show **before vs after** panel.

Good luck with the hackathon.
