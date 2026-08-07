"""
VLM Reviewer Data Types & Assessment Interfaces
================================================
Defines data structures for VLM event reviews, visual assessment verdicts,
and API request/response containers.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np


class VLMAssessmentVerdict(Enum):
    """Verdict result from VLM review of a suspicious event."""
    SUSPICIOUS = "SUSPICIOUS"
    BENIGN = "BENIGN"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class VLMAssessment:
    """Structured result returned from VLM analysis."""
    verdict: VLMAssessmentVerdict
    confidence: float
    reasoning: str
    detected_actions: List[str] = field(default_factory=list)
    risk_boost: float = 0.0
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMReviewRequest:
    """Container holding event metadata and image crop for VLM evaluation."""
    event_id: str
    track_id: int
    behavior_flag: str
    timestamp_ms: float
    frame: np.ndarray
    crop: Optional[np.ndarray] = None
    bbox: Optional[Any] = None
    context_text: Optional[str] = None
