from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from src.common.types import AlertLevel

@dataclass
class AlertMetadata:
    pre_event_frames: int
    post_event_frames: int
    raw_evidence_summary: str
    extra_details: Dict[str, Any]


@dataclass(frozen=True)
class Alert:
    id: str  # UUIDv4
    track_id: int
    camera_id: str
    timestamp_ms: float
    level: AlertLevel
    event_type: str
    clip_path: Optional[str] = None
    anonymized_image_path: Optional[str] = None
    metadata: Optional[AlertMetadata] = None
