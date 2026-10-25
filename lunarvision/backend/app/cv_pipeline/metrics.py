"""
Quality metrics for a registration result.

These are the numbers the problem statement (and any credible reviewer) will
actually check: how many correspondences were found, how many survived
geometric verification, and how well the estimated transform explains the
inlier points (RMSE of reprojection error, in pixels).

IMPORTANT: RMSE here is computed against the ESTIMATED homography using only
the inlier points that RANSAC itself selected. It reports how self-consistent
the estimated transform is - it is NOT ground-truth accuracy unless the pair
was synthetically generated with a KNOWN transform (see synthetic.py), in
which case we can additionally compare against ground truth.
"""

import numpy as np
import cv2


def compute_basic_metrics(matches, mask):
    total_matches = len(matches)
    if mask is None:
        return {
            "total_matches": total_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
    inliers = int(mask.sum())
    inlier_ratio = inliers / total_matches if total_matches > 0 else 0.0
    return {
        "total_matches": total_matches,
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
    }


def compute_rmse(src_pts, dst_pts, H, mask):
    """
    Reprojection RMSE (pixels) computed over INLIER points only:
        project src_pts through H, compare to dst_pts.
    """
    if H is None or mask is None:
        return None

    mask_flat = mask.ravel().astype(bool)
    if mask_flat.sum() == 0:
        return None

    src_inliers = src_pts[mask_flat]
    dst_inliers = dst_pts[mask_flat]

    src_h = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])  # Nx3
    projected = (H @ src_h.T).T  # Nx3
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(projected - dst_inliers, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return round(rmse, 4)


def compute_ground_truth_error(H_estimated, H_ground_truth, image_shape):
    """
    Only meaningful for SYNTHETIC pairs where the true transform is known.
    Compares how estimated H maps the four image corners vs. the true H maps
    them, and reports mean corner displacement error in pixels. This is the
    strongest honesty check we have available in the prototype.
    """
    if H_estimated is None or H_ground_truth is None:
        return None

    h, w = image_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    est_corners = cv2.perspectiveTransform(corners, H_estimated).reshape(-1, 2)
    gt_corners = cv2.perspectiveTransform(corners, H_ground_truth).reshape(-1, 2)

    errors = np.linalg.norm(est_corners - gt_corners, axis=1)
    return round(float(np.mean(errors)), 4)
