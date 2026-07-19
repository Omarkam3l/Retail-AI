"""Transforms, resizes, and normalizes image crops for PyTorch models."""
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesses image crops matching DINOv2 requirements."""

    def __init__(self, target_size: int = 224) -> None:
        self._target_size = target_size
        # ImageNet mean & std
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resizes, scales, normalizes, and reorders to CHW float32 format."""
        if crop.size == 0:
            return np.zeros((3, self._target_size, self._target_size), dtype=np.float32)

        # 1. Resize to target resolution (bilinear interpolation)
        resized = cv2.resize(crop, (self._target_size, self._target_size), interpolation=cv2.INTER_LINEAR)

        # 2. BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. 0-255 uint8 to 0-1 float32
        scaled = rgb.astype(np.float32) / 255.0

        # 4. Standard ImageNet Normalization
        normalized = (scaled - self._mean) / self._std

        # 5. HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def preprocess_batch(self, crops: list) -> np.ndarray:
        """Preprocesses a list of crops returning a single batch tensor array."""
        preprocessed = [self.preprocess(crop) for crop in crops]
        return np.stack(preprocessed, axis=0) if preprocessed else np.empty((0, 3, self._target_size, self._target_size), dtype=np.float32)
