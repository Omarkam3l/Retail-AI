import logging
from typing import List, Dict, Tuple
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord

logger = logging.getLogger("ConfusionMatrix")

class ConfusionMatrixGenerator:
    """Generates normalized confusion matrices for behavior classification."""

    def __init__(self, categories: List[str]) -> None:
        self._categories = categories
        # Include "none" for no-behavior
        self._labels = categories + ["none"]

    def generate(
        self,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord]
    ) -> Dict[str, Dict[str, int]]:
        """Returns a matrix as nested dict: matrix[actual][predicted] = count."""
        matrix: Dict[str, Dict[str, int]] = {}
        for actual in self._labels:
            matrix[actual] = {pred: 0 for pred in self._labels}

        gt_map = {gt.frame_index: gt for gt in ground_truths}
        pred_map = {p.frame_index: p for p in predictions}
        all_frames = set(gt_map.keys()) | set(pred_map.keys())

        for frame_idx in all_frames:
            gt = gt_map.get(frame_idx)
            pred = pred_map.get(frame_idx)

            gt_behaviors = set(gt.behaviors) if gt else set()
            pred_behaviors = set(pred.behaviors) if pred else set()

            for cat in self._categories:
                actual = cat if cat in gt_behaviors else "none"
                predicted = cat if cat in pred_behaviors else "none"
                matrix[actual][predicted] += 1

        return matrix

    def normalize(self, matrix: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """Normalizes confusion matrix rows to percentages."""
        normalized: Dict[str, Dict[str, float]] = {}
        for actual, row in matrix.items():
            total = sum(row.values())
            normalized[actual] = {}
            for predicted, count in row.items():
                normalized[actual][predicted] = count / total if total > 0 else 0.0
        return normalized

    def to_markdown(self, matrix: Dict[str, Dict[str, int]]) -> str:
        """Renders confusion matrix as a Markdown table."""
        headers = self._labels
        lines = ["| Actual \\ Predicted | " + " | ".join(headers) + " |"]
        lines.append("| --- | " + " | ".join(["---"] * len(headers)) + " |")
        for actual in headers:
            row = matrix.get(actual, {})
            cells = [str(row.get(pred, 0)) for pred in headers]
            lines.append(f"| {actual} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
