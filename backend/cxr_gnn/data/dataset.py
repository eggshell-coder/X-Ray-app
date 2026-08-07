"""
data/dataset.py — Image loading/preprocessing (inference subset).
"""

from __future__ import annotations
import io
from typing import Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage.transform import resize as sk_resize


def inspect_and_load_image(raw_bytes: bytes, size: int) -> Tuple[np.ndarray, dict]:
    """Loads image, converts to grayscale float32 [0,1], (size,size), and performs
    Layer-1 pre-checks (RGB color channel variance, aspect ratio, and anatomical
    tissue pixel distribution).

    Chest X-rays are monochromatic (R ≈ G ≈ B) with smooth anatomical mid-tones
    (lungs, ribs, heart). Diagrams, text documents, logos, and wallpapers have
    extreme black/white ratios and minimal mid-tones.

    Returns:
        (gray_img, check_dict)
    """
    is_color = False
    color_score = 0.0
    aspect_ratio = 1.0

    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))
        raw_img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            raw_img = raw_img.max() - raw_img
    except Exception:
        raw_img = imread(io.BytesIO(raw_bytes))

    if raw_img.ndim == 3:
        h, w = raw_img.shape[:2]
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        channels = raw_img[..., :3] if raw_img.shape[2] >= 3 else raw_img
        if channels.shape[2] == 3:
            # Normalize to 0..1 for color check
            c_norm = channels.astype(np.float32)
            if c_norm.max() > 1.0:
                c_norm /= 255.0

            # RGB channel pairwise difference variance
            diff_rg = np.abs(c_norm[..., 0] - c_norm[..., 1])
            diff_gb = np.abs(c_norm[..., 1] - c_norm[..., 2])
            color_score = float(np.mean(diff_rg + diff_gb))

            # Threshold: X-ray scans have color_score < 0.025. Colored photos/screenshots exceed 0.03.
            if color_score > 0.030:
                is_color = True

        img = rgb2gray(channels)
    else:
        h, w = raw_img.shape[:2]
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        img = raw_img.astype(np.float32)

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img = sk_resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    img = rescale_intensity(img, out_range=(0.0, 1.0)).astype(np.float32)

    # ── Anatomical Tissue Histogram Pre-check ─────────────────────────────
    p_black = float(np.mean(img < 0.03))        # pure black pixels (diagram backgrounds, wallpapers)
    p_white = float(np.mean(img > 0.97))        # pure white pixels (document text paper, banners)
    p_midtone = float(np.mean((img >= 0.12) & (img <= 0.88)))  # soft tissue & lungs

    is_diagram_or_text = False
    if p_midtone < 0.22 or p_black > 0.68 or p_white > 0.42:
        is_diagram_or_text = True

    passed = not is_color and (0.4 <= aspect_ratio <= 2.5) and not is_diagram_or_text

    reason = None
    if is_color:
        reason = "not_grayscale"
    elif not (0.4 <= aspect_ratio <= 2.5):
        reason = "invalid_aspect_ratio"
    elif is_diagram_or_text:
        reason = "non_xray_diagram_or_text"

    check_dict = {
        "is_color": is_color,
        "color_score": color_score,
        "aspect_ratio": aspect_ratio,
        "p_black": p_black,
        "p_white": p_white,
        "p_midtone": p_midtone,
        "passed": passed,
        "reason": reason,
    }

    return img, check_dict


def load_gray_bytes(raw_bytes: bytes, size: int) -> np.ndarray:
    """Legacy wrapper around inspect_and_load_image."""
    img, _ = inspect_and_load_image(raw_bytes, size)
    return img
