"""
Pluggable feature detectors.

SIFT  -> scale/rotation invariant, floating point descriptors, generally the
         strongest classical baseline for cross-modal/illumination-changed pairs.
ORB   -> binary descriptors, much faster, more sensitive to scale/illumination
         change but useful as a fast comparison baseline.

Both are exposed through the same interface so the rest of the pipeline
(matcher, registration) does not need to know which one is active.
"""

import cv2

SUPPORTED_DETECTORS = ("sift", "orb")


def get_detector(name: str):
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=4000)
    if name == "orb":
        return cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
    raise ValueError(f"Unsupported detector '{name}'. Supported: {SUPPORTED_DETECTORS}")


def detect_and_compute(gray_img, detector_name: str):
    """
    Returns (keypoints, descriptors).
    keypoints: list of cv2.KeyPoint
    descriptors: np.ndarray or None if nothing found
    """
    detector = get_detector(detector_name)
    kp, desc = detector.detectAndCompute(gray_img, None)
    return kp, desc
