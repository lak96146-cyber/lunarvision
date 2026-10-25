"""
Preprocessing utilities for lunar images.

Different Chandrayaan-2 sensors (OHRC, TMC-2, IIRS) produce images with very
different brightness/contrast characteristics, and even images from the SAME
sensor look different depending on Sun angle. CLAHE (Contrast Limited Adaptive
Histogram Equalization) is used here as a practical, well-established way to
normalize local contrast/illumination before feature detection - this is our
prototype-level answer to "Sun angle invariance". It does NOT fully solve
multi-modal invariance (that is a research problem), but it measurably helps
SIFT/ORB find stable keypoints across illumination differences.
"""

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img is None:
        raise ValueError("to_grayscale received None image")
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clahe_normalize(gray: np.ndarray, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    """Adaptive local contrast normalization - reduces impact of shadows / uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light denoising - lunar images (esp. real sensor data) often carry sensor noise."""
    return cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)


def resize_max_dim(img: np.ndarray, max_dim: int = 1024):
    """Resize so the longer side == max_dim (keeps aspect ratio). Returns (resized, scale_factor)."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= max_dim:
        return img, 1.0
    scale = max_dim / float(longer)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_pipeline(img: np.ndarray, max_dim: int = 1024, apply_denoise: bool = False):
    """
    Full preprocessing chain used before feature detection.
    Returns: (processed_gray, scale_factor_applied)
    """
    resized, scale = resize_max_dim(img, max_dim=max_dim)
    gray = to_grayscale(resized)
    if apply_denoise:
        gray = denoise(gray)
    normalized = clahe_normalize(gray)
    return normalized, scale
