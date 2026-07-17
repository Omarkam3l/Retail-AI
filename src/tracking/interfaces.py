from abc import abstractmethod
from typing import List, Tuple
import numpy as np
from src.common.interfaces import Lifecycle
from src.common.types import DetectedObject, TrackedPerson

class BaseTracker(Lifecycle):
    """Abstract base class defining the contract for Multi-Object Tracking algorithms."""

    @abstractmethod
    def track(
        self, 
        frame: np.ndarray, 
        detections: List[DetectedObject]
    ) -> Tuple[List[TrackedPerson], List[DetectedObject]]:
        """Associates object detections across frames to establish persistent paths.

        Args:
            frame: A NumPy array representing the raw video frame.
            detections: A list of objects detected in the current frame.

        Returns:
            A tuple containing:
            1. A list of active TrackedPerson instances containing persistent IDs.
            2. A list of active tracked carrying containers/items.

        Raises:
            InferenceError: If tracking association fails.
        """
        pass
