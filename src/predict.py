"""
Run inference on a single image: computes ELA, runs it through the trained
model, and prints whether the document looks authentic or forged.
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from ela import compute_ela_array

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model_smoke.keras")
IMG_SIZE = (128, 128)


def predict_image(image_path: str, model_path: str = MODEL_PATH):
    model = load_model(model_path)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(f"Prediction: {result['label']} ({result['confidence']}% confidence, raw score={result['raw_score']})")
