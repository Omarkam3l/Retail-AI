from abc import abstractmethod
from typing import List, Tuple
import numpy as np
from src.common.interfaces import Lifecycle
from src.common.types import FrameMetadata
from src.alerts.types import Alert

class BaseInferencePipeline(Lifecycle):
    """Abstract base class coordinating the edge-side computer vision inference cascade."""

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float
    ) -> Tuple[FrameMetadata, List[Alert]]:
        """Processes a single video frame through detection, tracking, pose, and alerts.

        Args:
            frame: A NumPy array representing the raw video frame.
            frame_index: Sequence count index of the frame.
            timestamp_ms: Epoch timestamp when the frame was captured.

        Returns:
            A tuple containing:
            1. The compiled FrameMetadata showing all tracks and keypoints.
            2. A list of generated Alerts, if any were triggered.

        Raises:
            PipelineError: If any pipeline stage fails.
        """
        pass
