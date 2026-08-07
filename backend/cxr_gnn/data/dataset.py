"""
data/dataset.py — Image loading/preprocessing (inference subset).
"""

from __future__ import annotations
import io

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage.transform import resize as sk_resize


def load_gray_bytes(raw_bytes: bytes, size: int) -> np.ndarray:
    """Same preprocessing as training's load_gray(), but from an in-memory
    upload instead of a file path: grayscale float32 [0,1], (size,size).
    Supports DICOM (.dcm) files as well as standard PNG/JPEG/BMP/TIFF images.
    """
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))
        img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img
    except Exception:
        img = imread(io.BytesIO(raw_bytes))

    if img.ndim == 3:
        img = img[..., :3] if img.shape[2] == 4 else img
        img = rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = sk_resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    img = rescale_intensity(img, out_range=(0.0, 1.0)).astype(np.float32)
    return img
