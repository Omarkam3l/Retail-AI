import logging
from typing import List, Dict
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, BehaviorMetricsResult

logger = logging.getLogger("BehaviorMetrics")

BEHAVIOR_CATEGORIES = [
    "pocket_concealment", "bag_concealment", "grab_and_leave",
    "loitering", "restricted_area_entry", "shelf_interaction"
]


class BehaviorMetricsCalculator:
    """Computes per-behavior Precision, Recall, F1, FP, FN."""

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> BehaviorMetricsResult:
        gt_map = {gt.frame_index: set(gt.behaviors) for gt in ground_truths}
        pred_map = {p.frame_index: set(p.behaviors) for p in predictions}
        all_frames = set(gt_map.keys()) | set(pred_map.keys())

        per_behavior: Dict[str, Dict[str, float]] = {}

        for behavior in BEHAVIOR_CATEGORIES:
            tp = fp = fn = 0
            for frame_idx in all_frames:
                gt_behaviors = gt_map.get(frame_idx, set())
                pred_behaviors = pred_map.get(frame_idx, set())

                gt_has = behavior in gt_behaviors
                pred_has = behavior in pred_behaviors

                if gt_has and pred_has:
                    tp += 1
                elif pred_has and not gt_has:
                    fp += 1
                elif gt_has and not pred_has:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_behavior[behavior] = {
                "precision": precision, "recall": recall, "f1": f1,
                "fp": float(fp), "fn": float(fn)
            }

        return BehaviorMetricsResult(per_behavior=per_behavior)
