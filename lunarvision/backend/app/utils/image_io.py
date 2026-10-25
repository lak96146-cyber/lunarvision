import numpy as np
import cv2


async def read_upload_as_bgr(upload_file):
    """Read a FastAPI UploadFile into an OpenCV BGR image."""
    contents = await upload_file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image file: {upload_file.filename}")
    return img
