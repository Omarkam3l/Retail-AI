from dataclasses import dataclass
from typing import Optional
from src.common.types import AssociationState, EventType, BoundingBox

@dataclass(frozen=True)
class AssociationEvent:
    event_type: EventType
    person_track_id: int
    object_track_id: int
    timestamp_ms: float
    confidence: float
    bbox: Optional[BoundingBox] = None


@dataclass
class AssociationMetadata:
    person_track_id: int
    object_track_id: int
    state: AssociationState
    confidence: float
    start_time_ms: float
    last_update_ms: float
    persistence_count: int = 1
    missed_count: int = 0
