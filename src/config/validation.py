from pydantic import BaseModel, Field, field_validator

class DetectorConfig(BaseModel):
    model_path: str = Field(..., description="Path to YOLO weights file")
    device: str = Field("cpu", description="Hardware device (cpu, cuda, mps)")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v


class TrackerConfig(BaseModel):
    track_threshold: float = Field(0.5, ge=0.0, le=1.0)
    match_threshold: float = Field(0.8, ge=0.0, le=1.0)
    track_buffer: int = Field(30, ge=1)


class AssociationConfig(BaseModel):
    proximity_threshold: float = Field(0.25, ge=0.0)
    persistence_threshold: int = Field(5, ge=1)
    lost_threshold: int = Field(30, ge=1)


class BehaviorConfig(BaseModel):
    max_sequence_gap_seconds: float = Field(10.0, ge=0.1)
    loiter_threshold_seconds: float = Field(15.0, ge=1.0)


class AppConfig(BaseModel):
    detector: DetectorConfig
    tracker: TrackerConfig
    association: AssociationConfig
    behavior: BehaviorConfig
