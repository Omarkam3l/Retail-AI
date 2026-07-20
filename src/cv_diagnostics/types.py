"""Shared data structures and types for the CV diagnostics module."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from src.common.types import BoundingBox, ClassLabel

@dataclass
class DetectionRecord:
    bbox: BoundingBox
    confidence: float
    class_label: str
    is_nms_suppressed: bool = False
    nms_suppression_reason: str = ""

@dataclass
class TrackRecord:
    track_id: int
    class_label: str
    bbox: BoundingBox
    confidence: float
    velocity: Tuple[float, float] = (0.0, 0.0)

@dataclass
class RecognitionRecord:
    track_id: int
    sku: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    similarity: float = 0.0
    confidence: float = 0.0

@dataclass
class AssociationRecord:
    person_track_id: int
    object_track_id: int
    confidence: float
    state: str

@dataclass
class FailureRecord:
    frame_index: int
    track_id: Optional[int]
    class_label: str
    category: str
    reason: str
    recommendation: str
    confidence: float
    bbox: BoundingBox
    crop_filename: str

@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    raw_frame: np.ndarray
    letterboxed_frame: np.ndarray
    detections_before_nms: List[DetectionRecord] = field(default_factory=list)
    detections_after_nms: List[DetectionRecord] = field(default_factory=list)
    tracks: List[TrackRecord] = field(default_factory=list)
    recognitions: List[RecognitionRecord] = field(default_factory=list)
    associations: List[AssociationRecord] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
