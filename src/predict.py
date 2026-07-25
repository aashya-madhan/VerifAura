"""
Run inference on a single image or multi-page document.

Functions:
  predict_image(path)                  — single image → {"label", "confidence", "raw_score"}
  predict_document(path, upload_dir)   — document   → per-page results + aggregate
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from ela import compute_ela_array

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model_smoke.keras")
IMG_SIZE = (128, 128)

# Cache the loaded model so it is not reloaded on every request
_model_cache: dict = {}


def _get_model(model_path: str):
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path)
    return _model_cache[model_path]


def predict_image(image_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Run ELA + CNN on a single image file.

    Returns:
        {
            "label":      "FORGED" | "AUTHENTIC",
            "confidence": float (0-100),
            "raw_score":  float (0-1),
        }
    """
    model = _get_model(model_path)
    arr = compute_ela_array(image_path, size=IMG_SIZE)
    arr = np.expand_dims(arr, axis=0)  # add batch dim

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "FORGED" if prob > 0.5 else "AUTHENTIC"
    confidence = prob if prob > 0.5 else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(prob, 4),
    }


def predict_document(filepath: str, upload_dir: str, model_path: str = MODEL_PATH) -> dict:
    """
    Analyse a document (PDF, DOCX, PPTX) or plain image.

    Each page / embedded image is individually run through ELA + CNN.
    An aggregate verdict is produced: if ANY page is flagged FORGED the
    whole document is marked FORGED.

    Args:
        filepath:   path to the uploaded file
        upload_dir: directory used to store extracted page images

    Returns:
        {
            "is_document":  bool,
            "file_type":    str,           e.g. "pdf", "docx", "jpg"
            "page_count":   int,
            "label":        "FORGED" | "AUTHENTIC",
            "confidence":   float (0-100),  # worst-case page confidence
            "raw_score":    float (0-1),    # highest raw forgery score across pages
            "pages": [
                {
                    "page_num":   int,
                    "image_path": str,        # path to the extracted page PNG
                    "ela_path":   str | None, # path to ELA visualisation PNG
                    "label":      str,
                    "confidence": float,
                    "raw_score":  float,
                },
                ...
            ]
        }
    """
    from document_processor import extract_pages, is_document
    from ela import compute_ela

    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    doc_flag = is_document(filepath)

    # Extract pages — returns list of (image_path, page_number)
    page_tuples = extract_pages(filepath, upload_dir)

    page_results = []
    for page_path, page_num in page_tuples:
        # Run prediction
        try:
            pred = predict_image(page_path, model_path)
        except Exception as e:
            pred = {
                "label": "ERROR",
                "confidence": 0.0,
                "raw_score": 0.0,
                "error": str(e),
            }

        # Save ELA visualisation for this page
        ela_path = None
        try:
            ela_filename = f"ela_{os.path.splitext(os.path.basename(page_path))[0]}.png"
            ela_full = os.path.join(upload_dir, ela_filename)
            compute_ela(page_path).save(ela_full)
            ela_path = ela_full
        except Exception:
            pass

        page_results.append({
            "page_num":   page_num,
            "image_path": page_path,
            "ela_path":   ela_path,
            "label":      pred["label"],
            "confidence": pred["confidence"],
            "raw_score":  pred["raw_score"],
        })

    # Aggregate: document is FORGED if any page score exceeds 0.5
    max_raw = max((p["raw_score"] for p in page_results), default=0.0)
    aggregate_label = "FORGED" if max_raw > 0.5 else "AUTHENTIC"
    worst_conf = max_raw if max_raw > 0.5 else 1 - max_raw

    return {
        "is_document": doc_flag,
        "file_type":   ext,
        "page_count":  len(page_results),
        "label":       aggregate_label,
        "confidence":  round(worst_conf * 100, 2),
        "raw_score":   round(max_raw, 4),
        "pages":       page_results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_document_path>")
        sys.exit(1)

    from document_processor import is_document
    path = sys.argv[1]

    if is_document(path):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = predict_document(path, tmpdir)
        print(f"\nDocument: {os.path.basename(path)}")
        print(f"Overall:  {result['label']} ({result['confidence']}% confidence)")
        print(f"Pages analysed: {result['page_count']}")
        for p in result["pages"]:
            print(f"  Page {p['page_num']:>3}: {p['label']} ({p['confidence']}%)")
    else:
        result = predict_image(path)
        print(
            f"Prediction: {result['label']} "
            f"({result['confidence']}% confidence, raw score={result['raw_score']})"
        )
