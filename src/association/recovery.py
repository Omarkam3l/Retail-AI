"""Association Recovery Engine for resolving lost track IDs using appearance and motion consistency."""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.common.types import BoundingBox, DetectedObject

logger = logging.getLogger("AssociationRecoveryEngine")


class AssociationRecoveryEngine:
    """Tracks inactive/lost objects and attempts to restore their identities upon re-detection."""

    def __init__(
        self,
        recovery_threshold: float = 0.82,
        max_recovery_age_frames: int = 45,
        max_spatial_distance: float = 0.35
    ) -> None:
        self._recovery_threshold = recovery_threshold
        self._max_age = max_recovery_age_frames
        self._max_dist = max_spatial_distance
        
        # Key: track_id -> dict with details (embedding, bbox, class_label, last_seen_frame, last_seen_time)
        self._history: Dict[int, dict] = {}

    def record_inactive(
        self,
        track_id: int,
        class_label: str,
        bbox: BoundingBox,
        embedding: np.ndarray,
        frame_index: int,
        timestamp_ms: float
    ) -> None:
        """Stores metadata for a lost/inactive track."""
        self._history[track_id] = {
            "class_label": class_label,
            "bbox": bbox,
            "embedding": embedding,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms
        }
        logger.info(f"Recorded inactive track for recovery: {track_id} (class: {class_label})")

    def attempt_recovery(
        self,
        detected_obj: DetectedObject,
        current_embedding: np.ndarray,
        frame_index: int,
        timestamp_ms: float
    ) -> Optional[int]:
        """Matches a new detection against inactive history. Returns the original track_id if recovered."""
        best_track_id: Optional[int] = None
        best_score = -1.0

        # Clean expired history
        self._history = {
            tid: data for tid, data in self._history.items()
            if frame_index - data["frame_index"] <= self._max_age
        }

        for tid, data in list(self._history.items()):
            if data["class_label"] != detected_obj.class_label:
                continue

            # 1. Cosine similarity
            hist_emb = data["embedding"]
            cos_sim = float(np.dot(hist_emb, current_embedding))

            # 2. Spatial distance
            p1 = data["bbox"].center
            p2 = detected_obj.bbox.center
            dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

            if dist > self._max_dist:
                continue

            # Temporal score decay
            time_decay = 1.0 - min(1.0, (timestamp_ms - data["timestamp_ms"]) / 5000.0)

            # Combined recovery score
            score = 0.6 * cos_sim + 0.3 * (1.0 - dist / self._max_dist) + 0.1 * time_decay

            if score > self._recovery_threshold and score > best_score:
                best_score = score
                best_track_id = tid

        if best_track_id is not None:
            logger.info(f"Successfully recovered track identity: {best_track_id} (Score: {best_score:.3f})")
            # Remove from recovery pool
            self._history.pop(best_track_id, None)

        return best_track_id

    def clear(self) -> None:
        self._history.clear()
