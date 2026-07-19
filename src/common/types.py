from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ClassLabel(str, Enum):
    PERSON = "person"
    BACKPACK = "backpack"
    HANDBAG = "handbag"
    SHOPPING_CART = "shopping_cart"
    SHOPPING_BASKET = "shopping_basket"
    SHELF_ITEM = "shelf_item"

class AlertLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EventType(str, Enum):
    PERSON_ENTERED_FRAME = "PERSON_ENTERED_FRAME"
    PERSON_LEFT_FRAME = "PERSON_LEFT_FRAME"
    PERSON_APPROACHES_EXIT = "PERSON_APPROACHES_EXIT"
    PERSON_ENTERS_RESTRICTED_AREA = "PERSON_ENTERS_RESTRICTED_AREA"
    PERSON_STATIONARY = "PERSON_STATIONARY"
    PRODUCT_PICKED = "PRODUCT_PICKED"
    PRODUCT_RETURNED = "PRODUCT_RETURNED"
    PRODUCT_DISAPPEARED = "PRODUCT_DISAPPEARED"
    HAND_NEAR_POCKET = "HAND_NEAR_POCKET"
    HAND_NEAR_BAG = "HAND_NEAR_BAG"
    PERSON_CROUCHES = "PERSON_CROUCHES"
    SHELF_INTERACTION = "SHELF_INTERACTION"

class AssociationState(str, Enum):
    UNASSOCIATED = "UNASSOCIATED"
    CANDIDATE = "CANDIDATE"
    ASSOCIATED = "ASSOCIATED"
    WEAK = "WEAK"
    LOST = "LOST"
    EXPIRED = "EXPIRED"

@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float

@dataclass(frozen=True)
class PoseKeypoints:
    keypoints: Dict[int, Keypoint] = field(default_factory=dict)

@dataclass(frozen=True)
class DetectedObject:
    class_label: ClassLabel
    bbox: BoundingBox
    confidence: float
    track_id: Optional[int] = None
    sku: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    similarity: Optional[float] = None
    rec_confidence: Optional[float] = None

@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: BoundingBox
    confidence: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    age_frames: int = 1
    keypoints: Optional[PoseKeypoints] = None

@dataclass(frozen=True)
class FrameMetadata:
    camera_id: str
    timestamp_ms: float
    frame_index: int
    persons: List[TrackedPerson] = field(default_factory=list)
    objects: List[DetectedObject] = field(default_factory=list)
