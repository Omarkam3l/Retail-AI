from dataclasses import dataclass
from enum import Enum
from typing import List, Any, Dict

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class Evidence:
    behavior_type: str
    confidence: float
    timestamp_ms: float
    raw_event: Any


@dataclass(frozen=True)
class RiskEvent:
    track_id: int
    risk_level: RiskLevel
    previous_level: RiskLevel
    score: float
    timestamp_ms: float
    evidence_list: List[Evidence]
