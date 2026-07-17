import cv2
import numpy as np
from src.common.types import BoundingBox

class FaceBlurProcessor:
    """Anonymizes shoppers' faces or body regions in frames to enforce GDPR compliance."""

    def __init__(self, kernel_size: int = 25) -> None:
        # Kernel size must be odd and positive
        self._ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def anonymize_region(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Applies Gaussian Blur to the bounding box region inside the frame."""
        h_frame, w_frame = frame.shape[:2]
        
        # Convert normalized coordinates to pixel indices
        x1 = max(0, int(bbox.x_min * w_frame))
        y1 = max(0, int(bbox.y_min * h_frame))
        x2 = min(w_frame, int(bbox.x_max * w_frame))
        y2 = min(h_frame, int(bbox.y_max * h_frame))

        if x2 <= x1 or y2 <= y1:
            return frame.copy()

        blurred_frame = frame.copy()
        roi = blurred_frame[y1:y2, x1:x2]
        
        # Apply heavy Gaussian blur to anonymize features
        blurred_roi = cv2.GaussianBlur(roi, (self._ksize, self._ksize), 0)
        blurred_frame[y1:y2, x1:x2] = blurred_roi
        
        return blurred_frame
