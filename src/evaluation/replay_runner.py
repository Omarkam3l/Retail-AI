import logging
import time
from typing import List, Optional, Callable, Any
from src.evaluation.types import GroundTruthAnnotation, PredictionRecord
from src.evaluation.prediction_logger import PredictionLogger

logger = logging.getLogger("ReplayEvaluationRunner")


class ReplayEvaluationRunner:
    """Offline deterministic replay engine for evaluation. Feeds ground-truth frame data
    through a pipeline callback and collects predictions."""

    def __init__(
        self,
        prediction_logger: Optional[PredictionLogger] = None,
        fixed_fps: float = 30.0
    ) -> None:
        self._logger = prediction_logger or PredictionLogger()
        self._fixed_fps = fixed_fps
        self._frame_interval_ms = 1000.0 / fixed_fps

    def run(
        self,
        ground_truths: List[GroundTruthAnnotation],
        pipeline_callback: Callable[[GroundTruthAnnotation], PredictionRecord],
        camera_id: str = "cam_eval"
    ) -> List[PredictionRecord]:
        """Runs the evaluation replay, passing each GT frame through the pipeline callback."""
        predictions: List[PredictionRecord] = []

        for i, gt in enumerate(ground_truths):
            # Assign deterministic timestamps
            gt_with_time = GroundTruthAnnotation(
                frame_index=gt.frame_index,
                timestamp_ms=gt.timestamp_ms if gt.timestamp_ms > 0 else i * self._frame_interval_ms,
                image_path=gt.image_path,
                bboxes=gt.bboxes,
                behaviors=gt.behaviors,
                risk_level=gt.risk_level,
                alerts=gt.alerts,
                metadata=gt.metadata
            )

            pred = pipeline_callback(gt_with_time)
            self._logger.log(pred)
            predictions.append(pred)

        logger.info(f"Replay completed: {len(predictions)} frames processed for camera '{camera_id}'.")
        return predictions
