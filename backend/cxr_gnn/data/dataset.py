"""
data/dataset.py — Image loading/preprocessing (inference subset).
"""

from __future__ import annotations
import io
from typing import Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import laplace
from skimage.io import imread
from skimage.transform import resize as sk_resize


def inspect_and_load_image(raw_bytes: bytes, size: int) -> Tuple[np.ndarray, dict]:
    """Loads image, converts to grayscale float32 [0,1], (size,size), and performs
    Layer 1 (Color/Tint Check), Layer 2 (Histogram/Gray-Level Check), and
    Layer 3 (Edge-Density Check).

    Allows:
      - Standard monochromatic chest X-rays.
      - Cyan/Blue/Sepia-tinted chest X-rays (uniform film/monitor color cast).

    Rejects:
      - Multi-colored non-medical photos (cats, nature, clothes, UI screenshots).
      - Flowchart diagrams, text screenshots, wallpapers.
      - Synthetic vector drawings / sharp document text.

    Returns:
        (gray_img, check_dict)
    """
    is_multi_color = False
    spatial_color_var = 0.0
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
            c_norm = channels.astype(np.float32)
            if c_norm.max() > 1.0:
                c_norm /= 255.0

            # Spatial Variance of RGB Color Ratios across pixels.
            # Uniform film/monitor tints (blue X-ray) have spatial_color_var < 0.015.
            # Multi-colored photos (cats, nature, UI screenshots) have spatial_color_var > 0.035.
            r, g, b = c_norm[..., 0], c_norm[..., 1], c_norm[..., 2]
            total = r + g + b + 1e-6
            r_ratio, g_ratio, b_ratio = r / total, g / total, b / total
            spatial_color_var = float(np.var(r_ratio) + np.var(g_ratio) + np.var(b_ratio))

            if spatial_color_var > 0.035:
                is_multi_color = True

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

    # ── Layer 2: Anatomical Tissue Histogram / Gray-Level Check ───────────
    p_black = float(np.mean(img < 0.03))        # pure black pixels (flowcharts, dark wallpapers)
    p_white = float(np.mean(img > 0.97))        # pure white pixels (document paper, text)
    p_midtone = float(np.mean((img >= 0.10) & (img <= 0.90)))  # soft tissue, lungs, heart

    is_diagram_or_text = False
    if p_midtone < 0.16 or p_black > 0.72 or p_white > 0.48:
        is_diagram_or_text = True

    # ── Layer 3: Edge Density Check (Detects text documents / vector lines)
    edges = laplace(img)
    edge_density = float(np.mean(np.abs(edges) > 0.15))
    is_dense_text_or_vector = False
    if edge_density > 0.35:
        is_dense_text_or_vector = True

    passed = (
        not is_multi_color
        and (0.35 <= aspect_ratio <= 2.8)
        and not is_diagram_or_text
        and not is_dense_text_or_vector
    )

    reason = None
    if is_multi_color:
        reason = "multi_colored_photo"
    elif not (0.35 <= aspect_ratio <= 2.8):
        reason = "invalid_aspect_ratio"
    elif is_diagram_or_text:
        reason = "non_xray_histogram"
    elif is_dense_text_or_vector:
        reason = "synthetic_edge_pattern"

    check_dict = {
        "is_multi_color": is_multi_color,
        "spatial_color_var": spatial_color_var,
        "aspect_ratio": aspect_ratio,
        "p_black": p_black,
        "p_white": p_white,
        "p_midtone": p_midtone,
        "edge_density": edge_density,
        "passed": passed,
        "reason": reason,
    }

    return img, check_dict


def load_gray_bytes(raw_bytes: bytes, size: int) -> np.ndarray:
    """Legacy wrapper around inspect_and_load_image."""
    img, _ = inspect_and_load_image(raw_bytes, size)
    return img
