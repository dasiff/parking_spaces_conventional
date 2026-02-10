"""Image preprocessing utilities for training and inference.

This module contains reusable helpers for image enhancement and normalization
that should be used consistently across training, inference, and visualization
code paths.

Public API:
- clahe_enhance_bgr(image_bgr, clip_limit=2.0, tile_grid_size=(8,8)) -> enhanced BGR image

The helpers are intentionally pure functions that accept and return numpy arrays.
"""
from typing import Tuple

import cv2
import numpy as np

__all__ = ["clahe_enhance_bgr"]


def clahe_enhance_bgr(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a BGR image and return the enhanced BGR image.

    Parameters
    ----------
    image_bgr: np.ndarray
        Input image in BGR color order (H,W,3), dtype=uint8.
    clip_limit: float
        CLAHE clip limit (default 2.0).
    tile_grid_size: Tuple[int,int]
        Tile grid size for CLAHE (default (8,8)).

    Returns
    -------
    np.ndarray
        Enhanced BGR image (same dtype as input).
    """
    if image_bgr is None:
        return image_bgr

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("clahe_enhance_bgr expects a HxWx3 BGR image")

    img = image_bgr.copy()

    # Convert to LAB color space and apply CLAHE to the L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return enhanced


if __name__ == "__main__":
    # simple smoke test
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = (50, 50, 80)  # a slightly bluish square
    out = clahe_enhance_bgr(img)
    assert out.shape == img.shape
    print("data_preprocessing: CLAHE smoke test OK")