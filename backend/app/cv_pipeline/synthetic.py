"""
Synthetic test-data generator.

THIS MODULE NEVER TOUCHES REAL CHANDRAYAAN-2 DATA.

It procedurally generates a crater/terrain-like grayscale texture (so the
pipeline has something visually lunar-ish to work with when real ISRO data
isn't available yet), then applies a KNOWN geometric + photometric transform
to create a second "sensor-like" view of the same scene. Because the
transform is known, we can report ground-truth error in addition to the
usual RANSAC self-consistency metrics - which is the strongest evidence we
can honestly show for a synthetic pair.

Every image and result produced from this module MUST be labeled
`is_synthetic = True` throughout the rest of the app.
"""

import cv2
import numpy as np


def generate_base_texture(size: int = 800, seed: int = None) -> np.ndarray:
    """Procedurally generate a crater-like grayscale terrain texture."""
    rng = np.random.default_rng(seed)

    # Multi-octave noise (cheap Perlin-ish effect via blurred random layers)
    canvas = np.zeros((size, size), dtype=np.float32)
    for octave, weight in [(4, 0.5), (8, 0.3), (16, 0.15), (32, 0.05)]:
        layer = rng.random((octave, octave)).astype(np.float32)
        layer = cv2.resize(layer, (size, size), interpolation=cv2.INTER_CUBIC)
        canvas += layer * weight

    canvas = cv2.normalize(canvas, None, 0, 200, cv2.NORM_MINMAX)

    # Stamp circular "craters" with rim highlight + shadow to mimic Sun-angle shading
    n_craters = rng.integers(15, 35)
    for _ in range(n_craters):
        cx, cy = rng.integers(0, size, size=2)
        r = rng.integers(8, 60)
        cv2.circle(canvas, (int(cx), int(cy)), int(r), color=-40, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(cx - r * 0.3), int(cy - r * 0.3)), int(r * 0.9), color=25, thickness=2)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def random_ground_truth_homography(image_shape, rotation_deg=None, scale=None,
                                    tx=None, ty=None, perspective_strength=0.0,
                                    seed: int = None):
    """
    Build a homography around the image center from simple, human-readable
    parameters (rotation, scale, translation, optional perspective wobble).
    Returns the 3x3 homography matrix (float64).
    """
    rng = np.random.default_rng(seed)
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    rotation_deg = rng.uniform(-25, 25) if rotation_deg is None else rotation_deg
    scale = rng.uniform(0.85, 1.15) if scale is None else scale
    tx = rng.uniform(-30, 30) if tx is None else tx
    ty = rng.uniform(-30, 30) if ty is None else ty

    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    H = np.vstack([M, [0, 0, 1]]).astype(np.float64)

    if perspective_strength > 0:
        # Small random perspective perturbation on top of the affine part
        p = perspective_strength
        H[2, 0] += rng.uniform(-p, p)
        H[2, 1] += rng.uniform(-p, p)

    return H


def apply_photometric_shift(img, brightness_delta=0, noise_sigma=0):
    """Apply brightness shift (illumination/Sun-angle proxy) and Gaussian noise (sensor noise proxy)."""
    out = img.astype(np.float32)
    out += brightness_delta
    if noise_sigma > 0:
        out += np.random.default_rng().normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_synthetic_pair(size: int = 800, rotation_deg=None, scale=None,
                             tx=None, ty=None, brightness_delta=None,
                             noise_sigma=None, perspective_strength=0.0,
                             seed: int = None):
    """
    Generate a (source, reference, H_ground_truth) triple.

    Convention used throughout this app: H_ground_truth maps SOURCE -> REFERENCE,
    matching the convention used by registration.estimate_homography().

    reference = base texture (as if it were, e.g., a TMC-2 view)
    source    = base texture warped + photometrically altered (as if a
                different sensor / different pass / different Sun angle)
    """
    rng = np.random.default_rng(seed)
    reference = generate_base_texture(size=size, seed=seed)

    brightness_delta = rng.uniform(-40, 40) if brightness_delta is None else brightness_delta
    noise_sigma = rng.uniform(2, 8) if noise_sigma is None else noise_sigma

    H_ref_to_src = random_ground_truth_homography(
        reference.shape, rotation_deg=rotation_deg, scale=scale, tx=tx, ty=ty,
        perspective_strength=perspective_strength, seed=seed,
    )
    # We generated H that maps reference->source layout conceptually via warping
    # reference by H's inverse convention; simplest: directly warp reference
    # using H_src_to_ref's inverse is confusing, so we instead define:
    # source = warp(reference, H_ref_to_src)  => this H maps REFERENCE coords to SOURCE image pixel positions.
    # For our pipeline convention (H maps SOURCE -> REFERENCE), ground truth is the inverse.
    h, w = reference.shape[:2]
    source = cv2.warpPerspective(reference, H_ref_to_src, (w, h), borderValue=30)
    source = apply_photometric_shift(source, brightness_delta=brightness_delta, noise_sigma=noise_sigma)

    H_source_to_reference = np.linalg.inv(H_ref_to_src)

    meta = {
        "rotation_deg": rotation_deg,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "brightness_delta": round(float(brightness_delta), 2),
        "noise_sigma": round(float(noise_sigma), 2),
        "perspective_strength": perspective_strength,
    }

    return source, reference, H_source_to_reference, meta
