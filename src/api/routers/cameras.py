"""Camera management endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from src.api.schemas import CameraInfo, CameraRegisterRequest, CameraActionResponse, CameraStatus
from src.api.dependencies import camera_registry
from typing import List

logger = logging.getLogger("CameraRouter")
router = APIRouter(tags=["Cameras"])


@router.get("/cameras", response_model=List[CameraInfo])
async def list_cameras() -> List[CameraInfo]:
    """Lists all registered cameras."""
    cameras = camera_registry.list_all()
    return [
        CameraInfo(
            camera_id=cid,
            source=info["source"],
            status=CameraStatus(info.get("status", "inactive")),
            fps=info.get("fps", 0.0),
            frame_count=info.get("frame_count", 0),
            resolution=info.get("resolution")
        )
        for cid, info in cameras.items()
    ]


@router.post("/camera/register", response_model=CameraActionResponse)
async def register_camera(request: CameraRegisterRequest) -> CameraActionResponse:
    """Registers a new camera source."""
    existing = camera_registry.get(request.camera_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Camera '{request.camera_id}' already registered.")

    camera_registry.register(
        camera_id=request.camera_id,
        source=request.source,
        confidence_threshold=request.confidence_threshold,
        device=request.device
    )

    return CameraActionResponse(
        camera_id=request.camera_id,
        action="register",
        success=True,
        message=f"Camera '{request.camera_id}' registered successfully."
    )


@router.post("/camera/start", response_model=CameraActionResponse)
async def start_camera(camera_id: str) -> CameraActionResponse:
    """Starts processing for a registered camera."""
    cam = camera_registry.get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    camera_registry.update_status(camera_id, "active")
    return CameraActionResponse(
        camera_id=camera_id,
        action="start",
        success=True,
        message=f"Camera '{camera_id}' started."
    )


@router.post("/camera/stop", response_model=CameraActionResponse)
async def stop_camera(camera_id: str) -> CameraActionResponse:
    """Stops processing for a registered camera."""
    cam = camera_registry.get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    camera_registry.update_status(camera_id, "inactive")
    return CameraActionResponse(
        camera_id=camera_id,
        action="stop",
        success=True,
        message=f"Camera '{camera_id}' stopped."
    )
