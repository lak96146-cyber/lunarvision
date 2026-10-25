"""
/api/analyze/*

- POST /api/analyze/synthetic-pair  -> generate a synthetic source/reference pair
- POST /api/analyze/run             -> run the full CV pipeline on two uploaded images
"""

import os
import time
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.config import SYNTHETIC_DIR, UPLOADS_DIR, PROCESSED_DIR, MAX_IMAGE_DIM
from app.cv_pipeline import synthetic, pipeline
from app.utils.image_io import read_upload_as_bgr

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

# In-memory cache of the last ground-truth homography per generated synthetic
# pair id, so /run can report ground-truth error if the user analyzes a
# synthetic pair we generated. Fine for a single-user local prototype.
_SYNTHETIC_GT_CACHE = {}


@router.post("/synthetic-pair")
def generate_synthetic_pair(
    rotation_deg: float = Form(None),
    scale: float = Form(None),
    tx: float = Form(None),
    ty: float = Form(None),
    brightness_delta: float = Form(None),
    noise_sigma: float = Form(None),
):
    pair_id = f"synth_{int(time.time() * 1000)}"
    source, reference, H_gt, meta = synthetic.generate_synthetic_pair(
        size=800, rotation_deg=rotation_deg, scale=scale, tx=tx, ty=ty,
        brightness_delta=brightness_delta, noise_sigma=noise_sigma,
    )
    source_path = os.path.join(SYNTHETIC_DIR, f"{pair_id}_source.png")
    reference_path = os.path.join(SYNTHETIC_DIR, f"{pair_id}_reference.png")
    cv2.imwrite(source_path, source)
    cv2.imwrite(reference_path, reference)

    _SYNTHETIC_GT_CACHE[pair_id] = H_gt.tolist()

    return {
        "pair_id": pair_id,
        "source_image": f"synthetic/{os.path.basename(source_path)}",
        "reference_image": f"synthetic/{os.path.basename(reference_path)}",
        "ground_truth_meta": meta,
        "is_synthetic": True,
    }


@router.post("/run")
async def run_analysis(
    source_sensor: str = Form(...),
    reference_sensor: str = Form(...),
    detector: str = Form("sift"),
    is_synthetic: bool = Form(False),
    synthetic_pair_id: str = Form(None),
    source_image: UploadFile = File(None),
    reference_image: UploadFile = File(None),
):
    detector = detector.lower()
    if detector not in ("sift", "orb"):
        raise HTTPException(400, "detector must be 'sift' or 'orb'")

    H_ground_truth = None

    if is_synthetic and synthetic_pair_id:
        # Load the previously generated synthetic pair from disk instead of
        # requiring re-upload.
        src_path = os.path.join(SYNTHETIC_DIR, f"{synthetic_pair_id}_source.png")
        ref_path = os.path.join(SYNTHETIC_DIR, f"{synthetic_pair_id}_reference.png")
        if not (os.path.exists(src_path) and os.path.exists(ref_path)):
            raise HTTPException(404, "synthetic_pair_id not found")
        source_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        reference_bgr = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        H_gt_list = _SYNTHETIC_GT_CACHE.get(synthetic_pair_id)
        if H_gt_list:
            H_ground_truth = np.array(H_gt_list)
    else:
        if source_image is None or reference_image is None:
            raise HTTPException(400, "source_image and reference_image are required when not using a synthetic pair")
        source_bgr = await read_upload_as_bgr(source_image)
        reference_bgr = await read_upload_as_bgr(reference_image)
        # Persist uploads for traceability
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(UPLOADS_DIR, f"{ts}_source.png"), source_bgr)
        cv2.imwrite(os.path.join(UPLOADS_DIR, f"{ts}_reference.png"), reference_bgr)

    file_prefix = f"run_{int(time.time() * 1000)}"
    result = pipeline.run_pipeline(
        source_bgr, reference_bgr, detector_name=detector,
        max_dim=MAX_IMAGE_DIM, H_ground_truth=H_ground_truth,
        processed_dir=PROCESSED_DIR, file_prefix=file_prefix,
    )

    result["source_sensor"] = source_sensor
    result["reference_sensor"] = reference_sensor
    result["is_synthetic"] = bool(is_synthetic)
    result["image_urls"] = {
        k: f"processed/{v}" for k, v in result["images"].items()
    }
    return result
