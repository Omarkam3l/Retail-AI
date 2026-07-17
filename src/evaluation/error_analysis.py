import logging
from typing import List, Dict
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, BBoxAnnotation, FailureCategory

logger = logging.getLogger("ErrorAnalysis")

class ErrorAnalyzer:
    """Classifies prediction failures into diagnostic categories."""

    def analyze(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> Dict[str, int]:
        """Returns counts of failures per category."""
        failure_counts: Dict[str, int] = {cat.value: 0 for cat in FailureCategory}

        gt_map = {gt.frame_index: gt for gt in ground_truths}
        pred_map = {p.frame_index: p for p in predictions}
        all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

        for frame_idx in all_frames:
            gt = gt_map.get(frame_idx)
            pred = pred_map.get(frame_idx)

            if gt is None or pred is None:
                continue

            gt_boxes = gt.bboxes
            pred_boxes = pred.detections

            # Missing detections: GT boxes with no matching prediction
            matched = set()
            for gb in gt_boxes:
                found = False
                for pi, pb in enumerate(pred_boxes):
                    if pi in matched:
                        continue
                    iou = self._compute_iou(gb, pb)
                    if iou >= 0.5 and gb.class_label == pb.class_label:
                        matched.add(pi)
                        found = True
                        break
                if not found:
                    # Classify failure based on metadata hints
                    category = self._classify_detection_failure(gt.metadata)
                    failure_counts[category] += 1

            # Tracking failures: track IDs present in GT but not in prediction
            gt_track_ids = {b.track_id for b in gt_boxes if b.track_id is not None}
            pred_track_ids = set(pred.track_ids) if pred.track_ids else set()
            missing_tracks = gt_track_ids - pred_track_ids
            if missing_tracks:
                failure_counts[FailureCategory.TRACKING_FAILURE.value] += len(missing_tracks)

            # Behavior failures
            gt_behaviors = set(gt.behaviors)
            pred_behaviors = set(pred.behaviors)
            missed_behaviors = gt_behaviors - pred_behaviors
            if missed_behaviors:
                failure_counts[FailureCategory.RULE_FAILURE.value] += len(missed_behaviors)

        return failure_counts

    def _classify_detection_failure(self, metadata: Dict) -> str:
        """Classifies detection failure based on metadata hints."""
        conditions = metadata.get("conditions", "")
        if "dark" in conditions or "night" in conditions:
            return FailureCategory.LIGHTING.value
        if "occluded" in conditions:
            return FailureCategory.OCCLUSION.value
        if "blur" in conditions:
            return FailureCategory.MOTION_BLUR.value
        if "angle" in conditions:
            return FailureCategory.CAMERA_ANGLE.value
        if "lowres" in conditions:
            return FailureCategory.LOW_RESOLUTION.value
        if "crowd" in conditions:
            return FailureCategory.CROWDED_SCENE.value
        return FailureCategory.MISSING_DETECTION.value

    def _compute_iou(self, a: BBoxAnnotation, b: BBoxAnnotation) -> float:
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
        area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
