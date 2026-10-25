"""
SQLite persistence layer.

One table: observations. Each row is a single source/reference analysis run
that the user chose to save from the Analyze page.
"""

import sqlite3
import json
import os
from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_sensor TEXT NOT NULL,
    reference_sensor TEXT NOT NULL,
    detector TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL,
    total_matches INTEGER,
    inliers INTEGER,
    inlier_ratio REAL,
    rmse_px REAL,
    ground_truth_corner_error_px REAL,
    homography_json TEXT,
    correspondence_image TEXT,
    warped_image TEXT,
    overlay_image TEXT,
    source_preprocessed TEXT,
    reference_preprocessed TEXT,
    notes TEXT
);
"""


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def insert_observation(data: dict) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO observations
        (created_at, source_sensor, reference_sensor, detector, is_synthetic,
         total_matches, inliers, inlier_ratio, rmse_px, ground_truth_corner_error_px,
         homography_json, correspondence_image, warped_image, overlay_image,
         source_preprocessed, reference_preprocessed, notes)
        VALUES (:created_at, :source_sensor, :reference_sensor, :detector, :is_synthetic,
                :total_matches, :inliers, :inlier_ratio, :rmse_px, :ground_truth_corner_error_px,
                :homography_json, :correspondence_image, :warped_image, :overlay_image,
                :source_preprocessed, :reference_preprocessed, :notes)
        """,
        data,
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_observations():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM observations ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_observation(obs_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_observations_by_ids(ids: list):
    conn = get_connection()
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM observations WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
