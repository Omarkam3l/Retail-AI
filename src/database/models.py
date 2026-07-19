"""Dataclasses representing database records."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertRecord:
    id: str
    track_id: int
    camera_id: str
    timestamp_ms: float
    level: str
    event_type: str
    clip_path: Optional[str] = None
    created_at: str = ""


@dataclass
class CameraRecord:
    camera_id: str
    source: str
    status: str = "inactive"
    fps: float = 0.0
    resolution: Optional[str] = None
    created_at: str = ""


@dataclass
class EventRecord:
    id: str
    camera_id: str
    event_type: str
    track_id: int
    timestamp_ms: float
    details: str = ""
    created_at: str = ""


@dataclass
class SystemLogRecord:
    id: str
    level: str
    module: str
    message: str
    created_at: str = ""


@dataclass
class BenchmarkRecord:
    id: str
    model_version: str
    dataset: str
    detection_f1: float = 0.0
    tracking_mota: float = 0.0
    execution_time_s: float = 0.0
    created_at: str = ""
