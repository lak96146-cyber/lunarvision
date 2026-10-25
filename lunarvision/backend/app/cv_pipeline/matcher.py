"""
Descriptor matching with Lowe's ratio test.

SIFT descriptors are float -> use L2 norm (FLANN or BF with NORM_L2).
ORB descriptors are binary -> must use Hamming distance (BF with NORM_HAMMING).

Lowe's ratio test discards ambiguous matches (where the best match is not
clearly better than the second-best), which is essential before RANSAC -
otherwise RANSAC has to fight through too much noise.
"""

import cv2

RATIO_TEST_THRESHOLD = 0.75


def match_descriptors(desc1, desc2, detector_name: str, ratio: float = RATIO_TEST_THRESHOLD):
    """
    Returns a list of good cv2.DMatch objects (query=desc1 index, train=desc2 index).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    detector_name = detector_name.lower()
    if detector_name == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif detector_name == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Unsupported detector for matching: {detector_name}")

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good
