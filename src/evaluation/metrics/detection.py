import logging
from typing import List, Tuple
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord, BBoxAnnotation, DetectionMetricsResult

logger = logging.getLogger("DetectionMetrics")


def compute_iou(box_a: BBoxAnnotation, box_b: BBoxAnnotation) -> float:
    """Computes Intersection over Union between two bounding boxes."""
    x1 = max(box_a.x_min, box_b.x_min)
    y1 = max(box_a.y_min, box_b.y_min)
    x2 = min(box_a.x_max, box_b.x_max)
    y2 = min(box_a.y_max, box_b.y_max)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min)
    area_b = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _match_detections(gt_boxes: List[BBoxAnnotation], pred_boxes: List[BBoxAnnotation],
                      iou_threshold: float) -> Tuple[int, int, int, List[float]]:
    """Greedy matching of predictions to ground truths at a given IoU threshold."""
    matched_gt = set()
    tp = 0
    ious: List[float] = []

    # Sort predictions by confidence descending
    sorted_preds = sorted(pred_boxes, key=lambda b: b.confidence, reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            if pred.class_label != gt.class_label:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, ious


class DetectionMetricsCalculator:
    """Computes detection evaluation metrics: Precision, Recall, F1, mAP50, mAP50-95."""

    def compute(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> DetectionMetricsResult:
        # Build lookup by frame_index
        gt_map = {gt.frame_index: gt.bboxes for gt in ground_truths}
        pred_map = {p.frame_index: p.detections for p in predictions}

        all_frames = set(gt_map.keys()) | set(pred_map.keys())

        # mAP across IoU thresholds [0.5, 0.55, ..., 0.95]
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        ap_per_threshold = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_ious: List[float] = []

        for iou_thresh in iou_thresholds:
            thresh_tp = 0
            thresh_fp = 0
            thresh_fn = 0

            for frame_idx in all_frames:
                gt_boxes = gt_map.get(frame_idx, [])
                pred_boxes = pred_map.get(frame_idx, [])
                tp, fp, fn, ious = _match_detections(gt_boxes, pred_boxes, iou_thresh)
                thresh_tp += tp
                thresh_fp += fp
                thresh_fn += fn

                if iou_thresh == 0.5:
                    all_ious.extend(ious)

            precision = thresh_tp / (thresh_tp + thresh_fp) if (thresh_tp + thresh_fp) > 0 else 0.0
            ap_per_threshold.append(precision)

            if iou_thresh == 0.5:
                total_tp = thresh_tp
                total_fp = thresh_fp
                total_fn = thresh_fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mAP50 = ap_per_threshold[0] if ap_per_threshold else 0.0
        mAP50_95 = sum(ap_per_threshold) / len(ap_per_threshold) if ap_per_threshold else 0.0

        return DetectionMetricsResult(
            precision=precision, recall=recall, f1=f1,
            mAP50=mAP50, mAP50_95=mAP50_95,
            iou_distribution=all_ious,
            num_true_positives=total_tp,
            num_false_positives=total_fp,
            num_false_negatives=total_fn
        )
