"""
/api/mosaic/*

Assumption made explicit here (and in the UI): mosaicking combines
observations that share the SAME reference image/frame (e.g. multiple source
images all registered against one common reference). This is the simplest
and most defensible way to combine multiple registrations into one regional
mosaic without introducing additional error from chaining unrelated
homographies together.
"""

import os
import time
import cv2
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import PROCESSED_DIR
from app import database
from app.mosaic_engine import stitcher

router = APIRouter(prefix="/api/mosaic", tags=["mosaic"])


class BuildMosaicRequest(BaseModel):
    observation_ids: list[int]


@router.post("/build")
def build_mosaic(req: BuildMosaicRequest):
    if len(req.observation_ids) < 1:
        raise HTTPException(400, "Provide at least one observation id")

    obs_list = database.get_observations_by_ids(req.observation_ids)
    if not obs_list:
        raise HTTPException(404, "No matching observations found")

    # preserve requested order
    obs_by_id = {o["id"]: o for o in obs_list}
    ordered = [obs_by_id[i] for i in req.observation_ids if i in obs_by_id]

    anchor_obs = ordered[0]
    anchor_ref_path = os.path.join(PROCESSED_DIR, anchor_obs["reference_preprocessed"])
    if not os.path.exists(anchor_ref_path):
        raise HTTPException(400, "Anchor observation's reference image file is missing on disk")
    anchor_ref_img = cv2.imread(anchor_ref_path, cv2.IMREAD_GRAYSCALE)

    pieces = []
    skipped = []
    for obs in ordered:
        src_path = os.path.join(PROCESSED_DIR, obs["source_preprocessed"]) if obs["source_preprocessed"] else None
        if not obs["homography_json"] or not src_path or not os.path.exists(src_path):
            skipped.append({"id": obs["id"], "reason": "missing homography or source image on disk"})
            continue
        H = np.array(json.loads(obs["homography_json"]))
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        pieces.append({"image": img, "H": H, "observation_id": obs["id"]})

    if not pieces:
        raise HTTPException(400, "None of the selected observations have usable homographies/images")

    mosaic_bgr, stats, contributing = stitcher.build_mosaic(anchor_ref_img, pieces)

    mosaic_filename = f"mosaic_{int(time.time() * 1000)}.png"
    mosaic_path = os.path.join(PROCESSED_DIR, mosaic_filename)
    cv2.imwrite(mosaic_path, mosaic_bgr)

    return {
        "mosaic_image": f"processed/{mosaic_filename}",
        "stats": stats,
        "anchor_observation_id": anchor_obs["id"],
        "contributing_observations": [o["id"] for o in ordered if o["id"] in [p["observation_id"] for p in pieces]],
        "skipped": skipped,
        "note": "Regional mosaic built from this app's own registrations. Not a complete/global lunar map.",
    }
