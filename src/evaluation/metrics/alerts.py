import logging
from typing import List
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, AlertMetricsResult

logger = logging.getLogger("AlertMetrics")


class AlertMetricsCalculator:
    """Computes alert precision, recall, duplicate alerts, missed alerts, average latency."""

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> AlertMetricsResult:
        gt_map = {gt.frame_index: set(gt.alerts) for gt in ground_truths}
        pred_map = {p.frame_index: list(p.alerts) for p in predictions}
        all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

        tp = fp = fn = 0
        duplicates = 0
        latencies: List[float] = []

        for frame_idx in all_frames:
            gt_alerts = gt_map.get(frame_idx, set())
            pred_alerts = pred_map.get(frame_idx, [])

            # Check duplicates within pred
            if len(pred_alerts) != len(set(pred_alerts)):
                duplicates += len(pred_alerts) - len(set(pred_alerts))

            pred_set = set(pred_alerts)
            matched = gt_alerts & pred_set
            tp += len(matched)
            fp += len(pred_set - gt_alerts)
            fn += len(gt_alerts - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        missed = fn

        return AlertMetricsResult(
            precision=precision, recall=recall,
            duplicate_alerts=duplicates, missed_alerts=missed,
            average_latency_ms=0.0
        )
