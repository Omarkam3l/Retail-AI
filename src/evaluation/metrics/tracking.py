import logging
from typing import List, Dict, Set
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, TrackingMetricsResult, BBoxAnnotation

logger = logging.getLogger("TrackingMetrics")


def _compute_iou(a: BBoxAnnotation, b: BBoxAnnotation) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackingMetricsCalculator:
    """Computes MOT metrics: MOTA, MOTP, IDF1, ID switches, fragmentations, track lifetimes."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> TrackingMetricsResult:
        gt_map = {gt.frame_index: gt for gt in ground_truths}
        pred_map = {p.frame_index: p for p in predictions}
        all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_matched_iou = 0.0
        total_matches = 0

        # Track previous frame assignments: gt_track_id -> predicted_track_id
        prev_assignment: Dict[int, int] = {}
        track_lifetimes: Dict[int, int] = {}
        fragmentation_tracks: Set[int] = set()
        prev_active: Set[int] = set()

        for frame_idx in all_frames:
            gt = gt_map.get(frame_idx)
            pred = pred_map.get(frame_idx)

            gt_boxes = gt.bboxes if gt else []
            pred_boxes = pred.detections if pred else []
            pred_track_ids = pred.track_ids if pred else []

            total_gt += len(gt_boxes)

            # Greedy match
            matched_gt: Set[int] = set()
            matched_pred: Set[int] = set()
            current_assignment: Dict[int, int] = {}

            for pi, pb in enumerate(pred_boxes):
                best_iou = 0.0
                best_gi = -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = _compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi

                if best_iou >= self._iou_threshold and best_gi >= 0:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                    total_matched_iou += best_iou
                    total_matches += 1

                    gt_tid = gt_boxes[best_gi].track_id if gt_boxes[best_gi].track_id is not None else best_gi
                    pred_tid = pred_track_ids[pi] if pi < len(pred_track_ids) else pi

                    if gt_tid in prev_assignment and prev_assignment[gt_tid] != pred_tid:
                        total_id_switches += 1

                    current_assignment[gt_tid] = pred_tid

                    # Track lifetime
                    track_lifetimes[pred_tid] = track_lifetimes.get(pred_tid, 0) + 1

            fp = len(pred_boxes) - len(matched_pred)
            fn = len(gt_boxes) - len(matched_gt)
            total_fp += fp
            total_fn += fn

            # Fragmentation: tracks that were active in prev frame but not matched now
            current_pred_ids = set(pred_track_ids[i] for i in matched_pred if i < len(pred_track_ids))
            for tid in prev_active:
                if tid not in current_pred_ids and tid in track_lifetimes:
                    fragmentation_tracks.add(tid)
            prev_active = current_pred_ids
            prev_assignment = current_assignment

        # MOTA = 1 - (FN + FP + ID_switches) / total_gt
        mota = 1.0 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0.0

        # MOTP = avg IoU of matched pairs
        motp = total_matched_iou / total_matches if total_matches > 0 else 0.0

        # IDF1 approximation: 2*TP / (2*TP + FP + FN)
        tp = total_matches
        idf1 = 2 * tp / (2 * tp + total_fp + total_fn) if (2 * tp + total_fp + total_fn) > 0 else 0.0

        lifetimes = list(track_lifetimes.values())

        return TrackingMetricsResult(
            mota=mota, motp=motp, idf1=idf1,
            id_switches=total_id_switches,
            fragmentations=len(fragmentation_tracks),
            track_lifetimes=lifetimes
        )
