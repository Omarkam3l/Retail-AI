from dataclasses import dataclass
from typing import Optional
from src.common.types import AlertLevel

@dataclass(frozen=True)
class Alert:
    id: str  # UUIDv4
    track_id: int
    camera_id: str
    timestamp_ms: float
    level: AlertLevel
    event_type: str
    clip_path: Optional[str] = None
