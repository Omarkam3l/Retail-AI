import logging
import time
from typing import List, Optional
from src.evaluation.types import (
    GroundTruthAnnotation, PredictionRecord, OverallEvaluationResult
)
from src.evaluation.metrics.detection import DetectionMetricsCalculator
from src.evaluation.metrics.tracking import TrackingMetricsCalculator
from src.evaluation.metrics.behavior import BehaviorMetricsCalculator
from src.evaluation.metrics.risk import RiskMetricsCalculator
from src.evaluation.metrics.alerts import AlertMetricsCalculator

logger = logging.getLogger("OverallMetrics")


class OverallMetricsCalculator:
    """Aggregates all metric calculators into a single evaluation result."""

    def __init__(
        self,
        compute_detection: bool = True,
        compute_tracking: bool = True,
        compute_behavior: bool = True,
        compute_risk: bool = True,
        compute_alerts: bool = True
    ) -> None:
        self._compute_detection = compute_detection
        self._compute_tracking = compute_tracking
        self._compute_behavior = compute_behavior
        self._compute_risk = compute_risk
        self._compute_alerts = compute_alerts

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> OverallEvaluationResult:
        start = time.perf_counter()
        result = OverallEvaluationResult()

        if self._compute_detection:
            result.detection = DetectionMetricsCalculator().compute(ground_truths, predictions)
        if self._compute_tracking:
            result.tracking = TrackingMetricsCalculator().compute(ground_truths, predictions)
        if self._compute_behavior:
            result.behavior = BehaviorMetricsCalculator().compute(ground_truths, predictions)
        if self._compute_risk:
            result.risk = RiskMetricsCalculator().compute(ground_truths, predictions)
        if self._compute_alerts:
            result.alerts = AlertMetricsCalculator().compute(ground_truths, predictions)

        result.execution_time_seconds = time.perf_counter() - start
        return result
