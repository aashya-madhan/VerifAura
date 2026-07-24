"""
Error Level Analysis (ELA) module.

ELA works by resaving an image at a known JPEG quality and computing the
pixel-wise difference against the original. Regions that were spliced/edited
tend to have different compression error characteristics than untouched
regions, because they were compressed a different number of times or at a
different quality before being pasted in.
"""

import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np


def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Compute the ELA image for a given input image.

    Args:
        image_path: path to the input image (any format PIL can open)
        quality: JPEG quality used for the resave step (default 90)
        scale: brightness amplification factor for the difference image

    Returns:
        PIL Image (RGB) representing the ELA output
    """
    original = Image.open(image_path).convert("RGB")

    # Resave at fixed JPEG quality into a temp buffer
    tmp_path = image_path + ".tmp_ela.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    resaved = Image.open(tmp_path)

    # Pixel-wise difference
    diff = ImageChops.difference(original, resaved)

    # Amplify the difference so subtle inconsistencies become visible
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    amplify = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(amplify)

    os.remove(tmp_path)
    return diff


def compute_ela_array(image_path: str, quality: int = 90, size: tuple = (128, 128)) -> np.ndarray:
    """
    Compute ELA and return as a normalized numpy array, resized for CNN input.

    Returns:
        np.ndarray of shape (H, W, 3), values in [0, 1]
    """
    ela_img = compute_ela(image_path, quality=quality)
    ela_img = ela_img.resize(size)
    arr = np.array(ela_img).astype("float32") / 255.0
    return arr


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ela.py <image_path> [output_path]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ela_output.png"

    ela_result = compute_ela(in_path)
    ela_result.save(out_path)
    print(f"ELA image saved to {out_path}")
