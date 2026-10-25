"""
Regional Lunar Mosaic builder.

IMPORTANT (per project scope): this produces a "Registered Lunar Mosaic /
Regional Lunar Map" from images the user has already registered and saved as
observations. It is NOT a claim of a complete/global scientifically accurate
Moon map - it is a regional composite built strictly from the homographies
this app itself estimated.

Approach (kept intentionally simple for the prototype):
  1. Take the FIRST observation's reference image as the anchor/base frame.
  2. For every observation, warp its (preprocessed) source image into the
     anchor frame using its stored homography, on a canvas large enough to
     contain all warped corners.
  3. Blend overlapping regions by simple averaging of non-black pixels.

This is a legitimate, explainable mosaicking method for a hackathon
prototype - it is not multi-band/Laplacian blending, but it is honest about
what it does.
"""

import cv2
import numpy as np
import json
import os


def _compute_canvas_bounds(shapes_and_homographies, anchor_shape):
    """Compute the bounding box (in anchor frame coords) needed to fit every
    warped image's corners, plus the anchor image itself."""
    h_anchor, w_anchor = anchor_shape[:2]
    all_corners = [np.float32([[0, 0], [w_anchor, 0], [w_anchor, h_anchor], [0, h_anchor]])]

    for shape, H in shapes_and_homographies:
        h, w = shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        all_corners.append(warped_corners)

    all_corners = np.vstack(all_corners)
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    return min_xy, max_xy


def build_mosaic(anchor_ref_img, pieces):
    """
    anchor_ref_img: grayscale/BGR image used as the base coordinate frame
                     (the reference image of the FIRST observation)
    pieces: list of dicts: {"image": np.ndarray (source, preprocessed gray),
                             "H": 3x3 np.ndarray mapping this source -> anchor's
                                  reference frame}

    Returns: mosaic_bgr (uint8), canvas_offset (x_off, y_off) so callers can
             map anchor-frame coordinates onto the final mosaic image.
    """
    shapes_and_h = [(p["image"].shape, p["H"]) for p in pieces]
    min_xy, max_xy = _compute_canvas_bounds(shapes_and_h, anchor_ref_img.shape)

    x_off, y_off = -min_xy[0], -min_xy[1]
    canvas_w = int(max_xy[0] - min_xy[0])
    canvas_h = int(max_xy[1] - min_xy[1])
    # Safety cap so a bad homography can't blow up memory
    canvas_w = min(canvas_w, 6000)
    canvas_h = min(canvas_h, 6000)

    translation = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]], dtype=np.float64)

    accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    def _to_gray(img):
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Place anchor reference image first
    anchor_gray = _to_gray(anchor_ref_img)
    H_anchor = translation
    warped_anchor = cv2.warpPerspective(anchor_gray, H_anchor, (canvas_w, canvas_h))
    mask_anchor = (warped_anchor > 0).astype(np.float32)
    accum += warped_anchor.astype(np.float32) * mask_anchor
    weight += mask_anchor

    contributing = [{"role": "anchor_reference"}]

    for p in pieces:
        H_full = translation @ p["H"]
        gray = _to_gray(p["image"])
        warped = cv2.warpPerspective(gray, H_full, (canvas_w, canvas_h))
        mask = (warped > 0).astype(np.float32)
        accum += warped.astype(np.float32) * mask
        weight += mask
        contributing.append({"observation_id": p.get("observation_id")})

    weight_safe = np.where(weight == 0, 1, weight)
    mosaic_gray = (accum / weight_safe).astype(np.uint8)
    mosaic_bgr = cv2.cvtColor(mosaic_gray, cv2.COLOR_GRAY2BGR)

    coverage_mask = (weight > 0)
    overlap_mask = (weight > 1)

    stats = {
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "coverage_pixels": int(coverage_mask.sum()),
        "overlap_pixels": int(overlap_mask.sum()),
        "num_pieces": len(pieces),
    }

    return mosaic_bgr, stats, contributing
