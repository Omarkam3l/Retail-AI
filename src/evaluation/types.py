from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class AnnotationFormat(Enum):
    COCO = "coco"
    YOLO = "yolo"
    CVAT = "cvat"
    LABEL_STUDIO = "label_studio"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FailureCategory(Enum):
    LIGHTING = "lighting"
    OCCLUSION = "occlusion"
    MOTION_BLUR = "motion_blur"
    CAMERA_ANGLE = "camera_angle"
    LOW_RESOLUTION = "low_resolution"
    CROWDED_SCENE = "crowded_scene"
    MISSING_DETECTION = "missing_detection"
    TRACKING_FAILURE = "tracking_failure"
    ASSOCIATION_FAILURE = "association_failure"
    POSE_FAILURE = "pose_failure"
    RULE_FAILURE = "rule_failure"


@dataclass
class BBoxAnnotation:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_label: str
    confidence: float = 1.0
    track_id: Optional[int] = None


@dataclass
class GroundTruthAnnotation:
    frame_index: int
    timestamp_ms: float
    image_path: str
    bboxes: List[BBoxAnnotation] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)
    risk_level: Optional[str] = None
    alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    frame_index: int
    timestamp_ms: float
    camera_id: str
    detections: List[BBoxAnnotation] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)
    risk_level: Optional[str] = None
    risk_score: float = 0.0
    alerts: List[str] = field(default_factory=list)


@dataclass
class DetectionMetricsResult:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    iou_distribution: List[float] = field(default_factory=list)
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0


@dataclass
class TrackingMetricsResult:
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    id_switches: int = 0
    fragmentations: int = 0
    track_lifetimes: List[int] = field(default_factory=list)


@dataclass
class BehaviorMetricsResult:
    per_behavior: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Each key -> {"precision": ..., "recall": ..., "f1": ..., "fp": ..., "fn": ...}


@dataclass
class RiskMetricsResult:
    precision: float = 0.0
    recall: float = 0.0
    average_delay_ms: float = 0.0
    escalation_accuracy: float = 0.0


@dataclass
class AlertMetricsResult:
    precision: float = 0.0
    recall: float = 0.0
    duplicate_alerts: int = 0
    missed_alerts: int = 0
    average_latency_ms: float = 0.0


@dataclass
class OverallEvaluationResult:
    detection: Optional[DetectionMetricsResult] = None
    tracking: Optional[TrackingMetricsResult] = None
    behavior: Optional[BehaviorMetricsResult] = None
    risk: Optional[RiskMetricsResult] = None
    alerts: Optional[AlertMetricsResult] = None
    execution_time_seconds: float = 0.0


@dataclass
class ExperimentRecord:
    experiment_id: str
    git_commit: str
    dataset_version: str
    model_version: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    metrics: Optional[OverallEvaluationResult] = None
    execution_time_seconds: float = 0.0
    notes: str = ""


@dataclass
class DatasetMetadata:
    name: str
    version: str
    num_images: int = 0
    num_annotations: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
