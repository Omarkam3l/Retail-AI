"""Inference endpoints for frame and video processing."""
import time
import logging
import base64
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.api.schemas import InferenceResult, AlertResponse
from src.api.dependencies import camera_registry, alert_store, metrics_store

logger = logging.getLogger("InferenceRouter")
router = APIRouter(tags=["Inference"])


@router.post("/infer/frame", response_model=InferenceResult)
async def infer_frame(
    file: UploadFile = File(...),
    camera_id: str = Form(default="default")
) -> InferenceResult:
    """Runs inference on a single uploaded frame image."""
    start = time.perf_counter()

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        import cv2
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Return placeholder result (pipeline integration would go here)
    return InferenceResult(
        frame_index=0,
        num_persons=0,
        num_objects=0,
        detections=[],
        tracks=[],
        behaviors=[],
        risk_level=None,
        alerts=[],
        latency_ms=latency_ms
    )


@router.post("/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    camera_id: str = Form(default="default")
) -> dict:
    """Uploads a video for batch inference processing."""
    import os
    import uuid
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    video_id = str(uuid.uuid4())[:8]
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(upload_dir, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    logger.info(f"Video uploaded: {filepath} ({len(contents)} bytes)")

    return {
        "video_id": video_id,
        "filename": filename,
        "size_bytes": len(contents),
        "status": "uploaded",
        "message": "Video uploaded successfully. Processing will begin shortly."
    }
