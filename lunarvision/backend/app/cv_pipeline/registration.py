"""
Geometric verification (RANSAC) + registration (homography estimation + warping).

Given raw candidate matches (which still contain outliers even after the ratio
test), RANSAC repeatedly samples minimal point sets, fits a homography, and
counts how many matches agree with it ("inliers"). The homography with the
most inlier support is kept. This is the standard classical approach to robust
model fitting under heavy outlier contamination, which is exactly our
situation with cross-sensor lunar image matches.
"""

import cv2
import numpy as np

MIN_MATCHES_FOR_HOMOGRAPHY = 4
DEFAULT_RANSAC_REPROJ_THRESHOLD = 5.0


def estimate_homography(kp1, kp2, matches, ransac_thresh: float = DEFAULT_RANSAC_REPROJ_THRESHOLD):
    """
    kp1: keypoints from source image
    kp2: keypoints from reference image
    matches: list of cv2.DMatch (query=kp1, train=kp2)

    Returns: (H, mask, src_pts, dst_pts)
        H: 3x3 homography mapping source -> reference, or None if not enough matches
        mask: Nx1 array of 0/1 (inlier flags), same order as `matches`
        src_pts, dst_pts: Nx2 float32 arrays of the point coordinates used
    """
    if len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    return H, mask, src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2)


def warp_source_to_reference(source_img, H, reference_shape):
    """Warp the source image into the reference image's coordinate frame."""
    h_ref, w_ref = reference_shape[:2]
    warped = cv2.warpPerspective(source_img, H, (w_ref, h_ref))
    return warped


def make_overlay(reference_img, warped_source_img, alpha: float = 0.5):
    """
    Blend reference (as-is) with the warped source to visually show alignment
    quality. Both images must already be the same size (reference's size).
    """
    if len(reference_img.shape) == 2:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    if len(warped_source_img.shape) == 2:
        warped_source_img = cv2.cvtColor(warped_source_img, cv2.COLOR_GRAY2BGR)

    # Tint reference red-ish and warped source green-ish so misalignment is
    # visually obvious (a classic "red/cyan" registration-check style view).
    ref_tinted = reference_img.copy()
    ref_tinted[:, :, 1] = 0
    ref_tinted[:, :, 0] = 0

    warp_tinted = warped_source_img.copy()
    warp_tinted[:, :, 2] = 0

    overlay = cv2.addWeighted(ref_tinted, alpha, warp_tinted, 1 - alpha, 0)
    return overlay
