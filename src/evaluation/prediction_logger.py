import json
import os
import logging
import threading
from typing import List
from src.evaluation.types import PredictionRecord

logger = logging.getLogger("PredictionLogger")


class PredictionLogger:
    """Thread-safe structured JSON logger for every prediction emitted by the pipeline."""

    def __init__(self, output_dir: str = "predictions") -> None:
        self._output_dir = output_dir
        self._records: List[PredictionRecord] = []
        self._lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)

    def log(self, record: PredictionRecord) -> None:
        with self._lock:
            self._records.append(record)

    def flush(self, run_id: str = "default") -> str:
        """Writes all buffered predictions to a JSON file and returns the path."""
        with self._lock:
            records = list(self._records)
            self._records.clear()

        output_path = os.path.join(self._output_dir, f"{run_id}_predictions.json")
        serialized = []
        for r in records:
            serialized.append({
                "frame_index": r.frame_index,
                "timestamp_ms": r.timestamp_ms,
                "camera_id": r.camera_id,
                "detections": [{"x_min": d.x_min, "y_min": d.y_min, "x_max": d.x_max,
                                "y_max": d.y_max, "class_label": d.class_label,
                                "confidence": d.confidence} for d in r.detections],
                "track_ids": r.track_ids,
                "behaviors": r.behaviors,
                "risk_level": r.risk_level,
                "risk_score": r.risk_score,
                "alerts": r.alerts
            })

        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)

        logger.info(f"Flushed {len(serialized)} prediction records to {output_path}.")
        return output_path

    def get_records(self) -> List[PredictionRecord]:
        with self._lock:
            return list(self._records)
