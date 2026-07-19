"""Drawer utility mapping SKU, brand, and matched details as bounding box overlays."""
import cv2
import numpy as np
from src.product_recognition.types import RecognitionResult
from src.common.types import BoundingBox


class ProductVisualizer:
    """Visualizes product recognition labels on the image frames."""

    @staticmethod
    def draw_labels(
        frame: np.ndarray,
        bbox: BoundingBox,
        result: RecognitionResult
    ) -> None:
        """Draws bounding box border and text labels for recognized/unknown objects."""
        h, w = frame.shape[:2]
        x1 = int(bbox.x_min * w)
        y1 = int(bbox.y_min * h)
        x2 = int(bbox.x_max * w)
        y2 = int(bbox.y_max * h)

        if result.recognized:
            color = (0, 255, 0)  # Green for recognized
            label = f"{result.brand} {result.name} ({result.confidence:.2f})"
            sublabel = f"SKU: {result.sku} | Sim: {result.similarity:.2f}"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"UNKNOWN ({result.confidence:.2f})"
            sublabel = f"Sim: {result.similarity:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw labels
        cv2.putText(frame, label, (x1, max(15, y1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, sublabel, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
