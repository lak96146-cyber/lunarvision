"""
Full correspondence + registration pipeline orchestrator.

This is the single function the API layer calls. It stitches together every
module in this package and produces both the numeric result and the saved
visualization images (correspondence drawing, registered overlay, warped
image) that the frontend displays.
"""

import os
import time
import cv2
import numpy as np

from app.cv_pipeline import preprocess, detectors, matcher, registration, metrics


def run_pipeline(source_bgr, reference_bgr, detector_name: str = "sift",
                  max_dim: int = 1024, H_ground_truth=None, processed_dir: str = None,
                  file_prefix: str = None):
    """
    source_bgr, reference_bgr: raw BGR images (as loaded by cv2.imread)
    detector_name: 'sift' or 'orb'
    H_ground_truth: optional 3x3 matrix (only for synthetic pairs), mapping
                     source(original scale) -> reference(original scale)
    processed_dir: folder to save visualization PNGs into
    file_prefix: filename prefix for saved images (e.g. an id/timestamp)

    Returns a dict with all metrics + relative file paths of saved images.
    """
    t0 = time.time()
    file_prefix = file_prefix or str(int(time.time() * 1000))

    src_gray, src_scale = preprocess.preprocess_pipeline(source_bgr, max_dim=max_dim)
    ref_gray, ref_scale = preprocess.preprocess_pipeline(reference_bgr, max_dim=max_dim)

    kp1, desc1 = detectors.detect_and_compute(src_gray, detector_name)
    kp2, desc2 = detectors.detect_and_compute(ref_gray, detector_name)

    good_matches = matcher.match_descriptors(desc1, desc2, detector_name)

    H, mask, src_pts, dst_pts = registration.estimate_homography(kp1, kp2, good_matches)

    basic = metrics.compute_basic_metrics(good_matches, mask)
    rmse = metrics.compute_rmse(src_pts, dst_pts, H, mask) if H is not None else None

    gt_error = None
    if H_ground_truth is not None and H is not None:
        # Ground truth was defined at ORIGINAL image scale; our detected H
        # operates at the (possibly downscaled) preprocessing scale, so we
        # rescale H back to original-image coordinates before comparing.
        H_full_scale = _rescale_homography(H, src_scale, ref_scale)
        gt_error = metrics.compute_ground_truth_error(H_full_scale, H_ground_truth, source_bgr.shape)

    result = {
        "detector": detector_name,
        "keypoints_source": len(kp1) if kp1 else 0,
        "keypoints_reference": len(kp2) if kp2 else 0,
        **basic,
        "rmse_px": rmse,
        "ground_truth_corner_error_px": gt_error,
        "homography": H.tolist() if H is not None else None,
        "processing_time_sec": round(time.time() - t0, 3),
        "registration_successful": H is not None,
    }

    image_paths = {}
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)

        # 1. Correspondence visualization (green=inlier, red=outlier)
        match_vis = _draw_matches_colored(src_gray, kp1, ref_gray, kp2, good_matches, mask)
        corr_path = os.path.join(processed_dir, f"{file_prefix}_correspondence.png")
        cv2.imwrite(corr_path, match_vis)
        image_paths["correspondence_image"] = os.path.basename(corr_path)

        if H is not None:
            warped = registration.warp_source_to_reference(src_gray, H, ref_gray.shape)
            warped_path = os.path.join(processed_dir, f"{file_prefix}_warped.png")
            cv2.imwrite(warped_path, warped)
            image_paths["warped_image"] = os.path.basename(warped_path)

            overlay = registration.make_overlay(ref_gray, warped)
            overlay_path = os.path.join(processed_dir, f"{file_prefix}_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            image_paths["overlay_image"] = os.path.basename(overlay_path)

        src_proc_path = os.path.join(processed_dir, f"{file_prefix}_source_preprocessed.png")
        ref_proc_path = os.path.join(processed_dir, f"{file_prefix}_reference_preprocessed.png")
        cv2.imwrite(src_proc_path, src_gray)
        cv2.imwrite(ref_proc_path, ref_gray)
        image_paths["source_preprocessed"] = os.path.basename(src_proc_path)
        image_paths["reference_preprocessed"] = os.path.basename(ref_proc_path)

    result["images"] = image_paths
    return result


def _rescale_homography(H, src_scale, ref_scale):
    """
    H was estimated on images resized by (src_scale, ref_scale) relative to
    originals. Rescale so H operates correctly on original-resolution coords:
        H_full = S_ref^-1 @ H @ S_src
    where S_x = diag(scale_x, scale_x, 1)
    """
    S_src = np.array([[src_scale, 0, 0], [0, src_scale, 0], [0, 0, 1]])
    S_ref_inv = np.array([[1 / ref_scale, 0, 0], [0, 1 / ref_scale, 0], [0, 0, 1]])
    return S_ref_inv @ H @ S_src


def _draw_matches_colored(img1, kp1, img2, kp2, matches, mask):
    """Custom colored match drawing: green = inlier, red = outlier/rejected."""
    mask_flat = mask.ravel().tolist() if mask is not None else [0] * len(matches)
    canvas = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=mask_flat if mask is not None else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if mask is not None:
        outlier_mask = [0 if m else 1 for m in mask_flat]
        canvas = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, canvas,
            matchColor=(0, 0, 255), singlePointColor=None,
            matchesMask=outlier_mask,
            flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    return canvas
