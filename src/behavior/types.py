from dataclasses import dataclass, field
from typing import List, Any, Optional
from src.common.types import EventType, BoundingBox

@dataclass(frozen=True)
class PrimitiveEvent:
    event_type: EventType
    track_id: int
    timestamp_ms: float
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    extra_metadata: Optional[Any] = None

@dataclass(frozen=True)
class BehaviorFlag:
    behavior_type: str
    track_id: int
    confidence: float
    timestamp_ms: float
    evidence_events: List[PrimitiveEvent] = field(default_factory=list)
