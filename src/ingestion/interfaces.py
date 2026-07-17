from abc import abstractmethod
from typing import Tuple, Optional
import numpy as np
from src.common.interfaces import Lifecycle

class VideoSource(Lifecycle):
    """Abstract base class defining the contract for all frame ingestion sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next video frame.

        Returns:
            A tuple containing:
            1. A boolean flag indicating success.
            2. The raw video frame as a NumPy array (BGR format), or None.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the video source is currently opened and available for reading."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Retrieves the native frame rate of the video source."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Retrieves the frame resolution of the video source (width, height)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases all open video resources and handles cleanup."""
        pass
