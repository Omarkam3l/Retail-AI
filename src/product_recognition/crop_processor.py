"""Safely crop bounding boxes from standard frames with validation and padding options."""
import logging
import cv2
import numpy as np
from src.common.types import BoundingBox

logger = logging.getLogger("CropProcessor")


class CropProcessor:
    """Safely extracts image crops corresponding to bounding boxes."""

    @staticmethod
    def extract_crop(frame: np.ndarray, bbox: BoundingBox, padding_pixels: int = 4) -> np.ndarray:
        """Extracts a bounding box crop from the frame with optional padding boundary checks."""
        h, w = frame.shape[:2]

        x1 = int(bbox.x_min * w)
        y1 = int(bbox.y_min * h)
        x2 = int(bbox.x_max * w)
        y2 = int(bbox.y_max * h)

        # Add padding
        x1 = max(0, x1 - padding_pixels)
        y1 = max(0, y1 - padding_pixels)
        x2 = min(w, x2 + padding_pixels)
        y2 = min(h, y2 + padding_pixels)

        # Boundary check
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop coordinates: x={x1}..{x2}, y={y1}..{y2}")
            return np.empty((0, 0, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        return crop.copy()
