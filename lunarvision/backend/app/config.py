import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DIR = os.path.join(DATA_DIR, "raw")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DB_PATH = os.path.join(BASE_DIR, "lunarvision.db")

SUPPORTED_SENSORS = ("OHRC", "TMC-2", "IIRS")
MAX_IMAGE_DIM = 1024

for d in (RAW_DIR, SYNTHETIC_DIR, UPLOADS_DIR, PROCESSED_DIR):
    os.makedirs(d, exist_ok=True)
