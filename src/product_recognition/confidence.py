"""Fuses detection confidence, image size, quality, and motion blur into final confidence metrics."""
import cv2
import numpy as np


class ConfidenceEngine:
    """Fuses multi-modal properties into a single unified recognition score."""

    def __init__(self, quality_threshold: float = 0.4) -> None:
        self._quality_threshold = quality_threshold

    def calculate_quality(self, crop: np.ndarray) -> float:
        """Estimates image quality from contrast and size."""
        if crop.size == 0:
            return 0.0

        h, w = crop.shape[:2]
        size_score = min(w * h / (128 * 128), 1.0)  # normalized around 128x128 resolution

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Lapalacian variance indicates focus level (motion blur estimator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)

        # Contrast score (standard deviation of grayscale values)
        contrast = gray.std()
        contrast_score = min(contrast / 100.0, 1.0)

        return 0.4 * size_score + 0.4 * blur_score + 0.2 * contrast_score

    def fuse_confidence(
        self,
        similarity: float,
        detection_confidence: float,
        crop: np.ndarray
    ) -> float:
        """Generates fused confidence (weighted score)."""
        quality = self.calculate_quality(crop)
        if quality < self._quality_threshold:
            # Degrade recognition confidence heavily on poor quality
            return float(similarity * 0.7)

        # Fused formula: 60% similarity + 20% detection confidence + 20% quality
        fused = 0.6 * similarity + 0.2 * detection_confidence + 0.2 * quality
        return float(np.clip(fused, 0.0, 1.0))
