"""
Walks data/raw and auto-detects CASIA2 folder structure.

CASIA2 uses these folder names inside data/raw/:
  Au/  (or CASIA2.0_Authentic, Au_jpg, etc.)  -> label 0  (authentic)
  Tp/  (or CASIA2.0_Tampered,  Tp_jpg, etc.)  -> label 1  (forged)

The script scans every subfolder of data/raw, matches it against known
CASIA2 prefixes, and falls back to the legacy 'authentic'/'forged' names
so old layouts still work without any renaming.

Output:
  data/processed/X.npy  - ELA arrays, shape (N, H, W, 3), float32
  data/processed/y.npy  - labels (0 = authentic, 1 = forged), int32

Smoke-test mode (default):
  Set SMOKE_TEST = True to cap each class at SMOKE_SAMPLES images and use
  a smaller SMOKE_IMG_SIZE. Flip to False for a full training run.
"""

import os
import random
import numpy as np
from tqdm import tqdm
from ela import compute_ela_array

# ---------------------------------------------------------------------------
# Smoke-test controls  <-- flip these before a full run
# ---------------------------------------------------------------------------
SMOKE_TEST    = False   # True = quick sanity check, False = full dataset
SMOKE_SAMPLES = 100    # images per class when SMOKE_TEST is True
SMOKE_IMG_SIZE = (64, 64)  # smaller resolution for the smoke run

# Full-run settings
IMG_SIZE   = (128, 128)
# ---------------------------------------------------------------------------

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

AUTHENTIC_PREFIXES = ("au",       "authentic", "casia2.0_authentic", "au_jpg")
FORGED_PREFIXES    = ("tp",       "forged",    "casia2.0_tampered",  "tp_jpg",
                      "tampered", "spliced")


def _detect_label(folder_name: str):
    """Return 0 (authentic), 1 (forged), or None (skip)."""
    lower = folder_name.lower()
    for prefix in AUTHENTIC_PREFIXES:
        if lower.startswith(prefix):
            return 0
    for prefix in FORGED_PREFIXES:
        if lower.startswith(prefix):
            return 1
    return None


def _collect_images(folder: str):
    """Recursively collect all image paths under *folder*."""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(root, f))
    return paths


def build_dataset():
    # Resolve active settings based on mode
    active_size    = SMOKE_IMG_SIZE if SMOKE_TEST else IMG_SIZE
    active_cap     = SMOKE_SAMPLES  if SMOKE_TEST else None
    mode_label     = f"SMOKE-TEST (cap={active_cap}, size={active_size})" \
                     if SMOKE_TEST else f"FULL RUN (size={active_size})"

    print(f"\n=== prepare_dataset  [{mode_label}] ===\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = [], []

    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    subfolders = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    if not subfolders:
        raise RuntimeError(
            f"No subfolders found in {RAW_DIR}.\n"
            "Place the CASIA2 'Au' and 'Tp' folders directly inside data/raw/."
        )

    label_map, skipped = {}, []
    for folder_name in subfolders:
        label = _detect_label(folder_name)
        if label is not None:
            label_map[folder_name] = label
        else:
            skipped.append(folder_name)

    if skipped:
        print(f"Skipping unrecognised folders: {skipped}")

    if not label_map:
        raise RuntimeError(
            f"None of the subfolders in {RAW_DIR} matched known CASIA2 names.\n"
            f"Found: {subfolders}\n"
            "Expected folders starting with: 'Au', 'Tp', 'authentic', or 'forged'."
        )

    class_labels = {0: "authentic", 1: "forged"}
    for folder_name, label in sorted(label_map.items(), key=lambda kv: kv[1]):
        folder_path = os.path.join(RAW_DIR, folder_name)
        images = _collect_images(folder_path)

        # Cap per-class sample count in smoke-test mode
        if active_cap and len(images) > active_cap:
            random.seed(42)
            images = random.sample(images, active_cap)

        print(f"[{class_labels[label]}] '{folder_name}' -> {len(images)} images")

        for fpath in tqdm(images, desc=folder_name):
            try:
                arr = compute_ela_array(fpath, size=active_size)
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"  Skipping {fpath}: {e}")

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")

    # Use separate filenames so smoke output never overwrites full data
    suffix = "_smoke" if SMOKE_TEST else ""
    np.save(os.path.join(OUT_DIR, f"X{suffix}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y{suffix}.npy"), y)

    print(f"\nDataset built: X={X.shape}, y={y.shape}")
    print(f"  Authentic: {(y == 0).sum()}")
    print(f"  Forged:    {(y == 1).sum()}")
    print(f"Saved to {OUT_DIR}  (suffix='{suffix}')")


if __name__ == "__main__":
    build_dataset()
