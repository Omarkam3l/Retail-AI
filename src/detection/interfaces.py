from abc import abstractmethod
from typing import List
import numpy as np
from src.common.interfaces import Lifecycle
from src.common.types import DetectedObject

class BaseDetector(Lifecycle):
    """Abstract base class defining the contract for object detection models."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Runs inference on a single raw video frame.

        Args:
            frame: A NumPy array representing the decoded video frame (BGR format).

        Returns:
            A list of DetectedObject instances containing bounding boxes,
            confidences, and object classifications.

        Raises:
            InferenceError: If model inference fails.
        """
        pass
