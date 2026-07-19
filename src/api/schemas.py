"""Pydantic request/response schemas for the REST API."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: float = 0.0


class CameraStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class CameraInfo(BaseModel):
    camera_id: str
    source: str
    status: CameraStatus = CameraStatus.INACTIVE
    fps: float = 0.0
    frame_count: int = 0
    resolution: Optional[str] = None


class CameraRegisterRequest(BaseModel):
    camera_id: str
    source: str  # file path, RTSP URL, or webcam index
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    device: str = "auto"


class CameraActionResponse(BaseModel):
    camera_id: str
    action: str
    success: bool
    message: str = ""


class AlertResponse(BaseModel):
    id: str
    track_id: int
    camera_id: str
    timestamp_ms: float
    level: str
    event_type: str
    clip_path: Optional[str] = None


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]
    total: int
    page: int = 1
    page_size: int = 50


class InferFrameRequest(BaseModel):
    camera_id: str = "default"


class InferenceResult(BaseModel):
    frame_index: int
    num_persons: int
    num_objects: int
    detections: List[Dict[str, Any]]
    tracks: List[int]
    behaviors: List[str]
    risk_level: Optional[str] = None
    alerts: List[AlertResponse]
    latency_ms: float


class SystemStatusResponse(BaseModel):
    status: str = "operational"
    active_cameras: int = 0
    total_alerts: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_available: bool = False
    gpu_memory_percent: float = 0.0
    uptime_seconds: float = 0.0


class MetricsResponse(BaseModel):
    pipeline_fps: float = 0.0
    avg_latency_ms: float = 0.0
    total_frames_processed: int = 0
    total_detections: int = 0
    total_alerts: int = 0
    stage_latencies: Dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "INTERNAL_ERROR"
    timestamp: str = ""
