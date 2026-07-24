# Document Forgery Detection — ELA + CNN

Detects whether a document/image has been digitally forged by combining
**Error Level Analysis (ELA)** preprocessing with a **CNN classifier**.

## How it works

1. **ELA preprocessing** (`src/ela.py`) — resave the image at a fixed JPEG
   quality, diff it against the original, and amplify the result. Regions
   that were spliced/edited show a different error level than untouched
   regions because they've been compressed a different number of times.
2. **CNN classifier** (`src/model.py`) — trained on ELA images (not raw
   images) to classify authentic vs. forged.

## Project structure

```
doc-forgery-ela-cnn/
├── data/
│   ├── raw/
│   │   ├── authentic/     # put real/untouched images here
│   │   └── forged/        # put tampered/spliced images here
│   └── processed/         # generated X.npy / y.npy (after prepare_dataset.py)
├── models/                # trained model weights + training curves
├── src/
│   ├── ela.py             # ELA computation
│   ├── prepare_dataset.py # builds ELA dataset from data/raw
│   ├── model.py           # CNN architecture
│   ├── train.py           # training loop
│   ├── predict.py         # single-image inference (CLI)
│   └── app.py             # Flask web UI + API
├── static/uploads/        # runtime uploads for the web app
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get a dataset

Recommended: **CASIA v2.0** (standard image forgery detection dataset,
~12,600 images). Search "CASIA v2 dataset download" — it's freely available
for research use. Extract it so you have:

```
data/raw/authentic/*.jpg
data/raw/forged/*.jpg
```

CASIA is natural images, not scanned documents — if your real goal is
specifically scanned/photographed *documents* (IDs, certificates, invoices),
supplement with your own dataset: take real documents, then create forged
copies by editing text/fields with GIMP/Photoshop and re-saving as JPEG.
The forged/authentic split matters more than the subject matter for ELA to
learn the compression-artifact signal.

## 2. Build the ELA dataset

```bash
cd src
python prepare_dataset.py
```

This walks `data/raw/authentic` and `data/raw/forged`, computes ELA for
every image, and saves `data/processed/X.npy` and `data/processed/y.npy`.

## 3. Train

```bash
python train.py
```

Trains with an 70/15/15 train/val/test split, early stopping, and saves:
- `models/best_model.keras` (best val-loss checkpoint)
- `models/final_model.keras`
- `models/training_curves.png`
- prints a classification report + confusion matrix on the test set

## 4. Predict on a single image (CLI)

```bash
python predict.py /path/to/image.jpg
```

## 5. Run the web app

```bash
python app.py
```

Visit `http://localhost:5000`, upload an image, get a forged/authentic
verdict with confidence score plus the ELA visualization.

## Tuning notes

- **JPEG quality for ELA** (default 90): lower values increase sensitivity
  but also increase noise. Try 85–95 and see what separates your classes
  best.
- **Class imbalance**: forgery datasets are often skewed. Watch precision/
  recall, not just accuracy — a model that predicts "authentic" every time
  can still score high accuracy on an imbalanced set.
- **Overfitting to JPEG artifacts**: validate on images re-saved at
  different qualities so the model learns forgery patterns, not just "this
  image was resaved at quality X."
- **Transfer learning option**: `model.py` includes `build_cnn_transfer()`
  using a frozen MobileNetV2 backbone — useful if your labeled dataset is
  small (a few hundred images rather than thousands).

## Extending this project

- Add Grad-CAM to visualize *which region* of the ELA image triggered the
  forged classification (useful for a demo/report).
- Add multi-class output (splicing vs. copy-move vs. authentic) if your
  dataset has those labels.
- Add PDF support: rasterize each PDF page to an image before running ELA.
