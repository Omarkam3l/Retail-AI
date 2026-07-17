import logging
from typing import List
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, RiskMetricsResult

logger = logging.getLogger("RiskMetrics")


class RiskMetricsCalculator:
    """Computes risk precision, recall, average delay, and escalation accuracy."""

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> RiskMetricsResult:
        gt_map = {gt.frame_index: gt for gt in ground_truths}
        pred_map = {p.frame_index: p for p in predictions}
        all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

        tp = fp = fn = 0
        correct_escalations = 0
        total_escalations = 0
        delays: List[float] = []

        for frame_idx in all_frames:
            gt = gt_map.get(frame_idx)
            pred = pred_map.get(frame_idx)
            gt_risk = gt.risk_level if gt else None
            pred_risk = pred.risk_level if pred else None

            gt_has_risk = gt_risk is not None and gt_risk not in ("", "LOW", "NONE")
            pred_has_risk = pred_risk is not None and pred_risk not in ("", "LOW", "NONE")

            if gt_has_risk and pred_has_risk:
                tp += 1
                if gt_risk == pred_risk:
                    correct_escalations += 1
                total_escalations += 1
            elif pred_has_risk and not gt_has_risk:
                fp += 1
            elif gt_has_risk and not pred_has_risk:
                fn += 1

            # Delay: difference in timestamps between GT risk and pred risk
            if gt_has_risk and pred_has_risk and gt and pred:
                delay = abs(pred.timestamp_ms - gt.timestamp_ms)
                delays.append(delay)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        escalation_acc = correct_escalations / total_escalations if total_escalations > 0 else 0.0

        return RiskMetricsResult(
            precision=precision, recall=recall,
            average_delay_ms=avg_delay, escalation_accuracy=escalation_acc
        )
