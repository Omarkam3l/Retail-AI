"""Captures and saves JPEG snapshots at alert time."""
import os
import logging
import json
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger("SnapshotManager")


class SnapshotManager:
    """Captures and saves JPEG snapshots."""

    def __init__(self, output_dir: str = "data/snapshots", quality: int = 95) -> None:
        self._output_dir = output_dir
        self._quality = quality
        os.makedirs(output_dir, exist_ok=True)

    def capture(self, frame: np.ndarray, snapshot_id: str,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Saves a frame as a JPEG snapshot."""
        import cv2
        snapshot_path = os.path.join(self._output_dir, f"{snapshot_id}.jpg")
        cv2.imwrite(snapshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])

        if metadata:
            meta_path = os.path.join(self._output_dir, f"{snapshot_id}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Snapshot saved: {snapshot_path}")
        return snapshot_path
