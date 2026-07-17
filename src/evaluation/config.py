from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class DatasetConfig(BaseModel):
    name: str
    root_path: str
    format: str = "coco"
    version: str = "1.0"


class MetricsConfig(BaseModel):
    compute_detection: bool = True
    compute_tracking: bool = True
    compute_behavior: bool = True
    compute_risk: bool = True
    compute_alerts: bool = True
    iou_thresholds: List[float] = Field(default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])


class ThresholdSearchConfig(BaseModel):
    detection_confidences: List[float] = Field(default=[0.3, 0.4, 0.5, 0.6, 0.7])
    behavior_confidences: List[float] = Field(default=[0.4, 0.5, 0.6, 0.7])
    risk_thresholds: List[str] = Field(default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    objective: str = "f1"


class ReportConfig(BaseModel):
    output_dir: str = "reports"
    formats: List[str] = Field(default=["markdown"])
    include_charts: bool = True
    include_confusion_matrices: bool = True


class EvaluationConfig(BaseModel):
    datasets: List[DatasetConfig] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    thresholds: ThresholdSearchConfig = Field(default_factory=ThresholdSearchConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
