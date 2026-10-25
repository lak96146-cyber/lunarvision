"""
/api/observations/*
"""

import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app import database

router = APIRouter(prefix="/api/observations", tags=["observations"])


class SaveObservationRequest(BaseModel):
    source_sensor: str
    reference_sensor: str
    detector: str
    is_synthetic: bool
    total_matches: int
    inliers: int
    inlier_ratio: float
    rmse_px: Optional[float] = None
    ground_truth_corner_error_px: Optional[float] = None
    homography: Optional[list] = None
    correspondence_image: Optional[str] = None
    warped_image: Optional[str] = None
    overlay_image: Optional[str] = None
    source_preprocessed: Optional[str] = None
    reference_preprocessed: Optional[str] = None
    notes: Optional[str] = ""


@router.post("/save")
def save_observation(req: SaveObservationRequest):
    data = req.dict()
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["is_synthetic"] = int(data["is_synthetic"])
    data["homography_json"] = json.dumps(data.pop("homography"))
    new_id = database.insert_observation(data)
    return {"id": new_id, "status": "saved"}


@router.get("")
def list_observations():
    rows = database.list_observations()
    for r in rows:
        r["homography"] = json.loads(r.pop("homography_json")) if r.get("homography_json") else None
        r["is_synthetic"] = bool(r["is_synthetic"])
    return rows


@router.get("/{obs_id}")
def get_observation(obs_id: int):
    row = database.get_observation(obs_id)
    if not row:
        raise HTTPException(404, "Observation not found")
    row["homography"] = json.loads(row.pop("homography_json")) if row.get("homography_json") else None
    row["is_synthetic"] = bool(row["is_synthetic"])
    return row
