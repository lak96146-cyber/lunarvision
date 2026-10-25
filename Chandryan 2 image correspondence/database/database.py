import sqlite3
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    BASE_DIR,
    "instance",
    "lunarvision.db"
)


def get_connection():
    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True
    )

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            job_id TEXT NOT NULL,

            source_sensor TEXT NOT NULL,

            reference_sensor TEXT NOT NULL,

            source_keypoints INTEGER DEFAULT 0,

            reference_keypoints INTEGER DEFAULT 0,

            total_matches INTEGER DEFAULT 0,

            inlier_matches INTEGER DEFAULT 0,

            inlier_ratio REAL DEFAULT 0,

            rmse REAL,

            status TEXT,

            correspondence_image TEXT,

            registered_image TEXT,

            overlay_image TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_observation(data):

    conn = get_connection()

    cursor = conn.execute("""
        INSERT INTO observations (

            job_id,
            source_sensor,
            reference_sensor,
            source_keypoints,
            reference_keypoints,
            total_matches,
            inlier_matches,
            inlier_ratio,
            rmse,
            status,
            correspondence_image,
            registered_image,
            overlay_image

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        data["job_id"],
        data["source_sensor"],
        data["reference_sensor"],
        data["source_keypoints"],
        data["reference_keypoints"],
        data["total_matches"],
        data["inlier_matches"],
        data["inlier_ratio"],
        data["rmse"],
        data["status"],
        data["match_visual"],
        data["registered_visual"],
        data["overlay_visual"]
    ))

    conn.commit()

    observation_id = cursor.lastrowid

    conn.close()

    return observation_id


def get_all_observations():

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM observations
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return rows


def get_observation(observation_id):

    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM observations
        WHERE id = ?
    """, (observation_id,)).fetchone()

    conn.close()

    return row