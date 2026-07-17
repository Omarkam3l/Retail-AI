import logging
from typing import List, Dict, Callable, Tuple, Any
from src.evaluation.types import (
    GroundTruthAnnotation, PredictionRecord, BBoxAnnotation,
    DetectionMetricsResult
)
from src.evaluation.metrics.detection import DetectionMetricsCalculator

logger = logging.getLogger("ThresholdOptimizer")

class ThresholdOptimizer:
    """Grid search over detection/behavior/risk thresholds to find optimal configuration."""

    def __init__(
        self,
        detection_confidences: List[float] = None,
        behavior_confidences: List[float] = None,
        risk_thresholds: List[str] = None,
        objective: str = "f1"
    ) -> None:
        self._det_confs = detection_confidences or [0.3, 0.4, 0.5, 0.6, 0.7]
        self._beh_confs = behavior_confidences or [0.4, 0.5, 0.6, 0.7]
        self._risk_thresholds = risk_thresholds or ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self._objective = objective

    def optimize(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> Dict[str, Any]:
        """Runs grid search and returns the optimal threshold configuration."""
        best_score = -1.0
        best_config: Dict[str, Any] = {}
        all_results: List[Dict[str, Any]] = []

        calc = DetectionMetricsCalculator()

        for det_conf in self._det_confs:
            # Filter predictions by detection confidence
            filtered_preds = []
            for pred in predictions:
                filtered_dets = [d for d in pred.detections if d.confidence >= det_conf]
                filtered_preds.append(PredictionRecord(
                    frame_index=pred.frame_index,
                    timestamp_ms=pred.timestamp_ms,
                    camera_id=pred.camera_id,
                    detections=filtered_dets,
                    track_ids=pred.track_ids,
                    behaviors=pred.behaviors,
                    risk_level=pred.risk_level,
                    risk_score=pred.risk_score,
                    alerts=pred.alerts
                ))

            result = calc.compute(ground_truths, filtered_preds)

            score = getattr(result, self._objective, result.f1)

            entry = {
                "detection_confidence": det_conf,
                "score": score,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1
            }
            all_results.append(entry)

            if score > best_score:
                best_score = score
                best_config = {
                    "detection_confidence": det_conf,
                    "best_score": score,
                    "metrics": entry
                }

        logger.info(f"Threshold optimization complete. Best {self._objective}: {best_score:.4f}")
        return {
            "best": best_config,
            "all_results": all_results
        }
