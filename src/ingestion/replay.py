from typing import Tuple, Optional, Callable
import cv2
import time
import numpy as np
from src.ingestion.exceptions import VideoSourceError

class VideoReplayEngine:
    """Offline validation engine to step through or play pre-recorded video files at controlled rates."""

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

    def initialize(self) -> None:
        if self._cap is not None:
            self.release()
        self._cap = cv2.VideoCapture(self._file_path)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Failed to open replay file: {self._file_path}")
        self._frame_index = 0

    def next_frame(self) -> Tuple[bool, Optional[np.ndarray], dict]:
        """Steps forward and returns the next frame with metadata."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, {}

        success, frame = self._cap.read()
        if not success or frame is None:
            return False, None, {}

        metadata = {
            "frame_index": self._frame_index,
            "timestamp_ms": time.time() * 1000.0,
            "replay_file": self._file_path
        }
        self._frame_index += 1
        return True, frame, metadata

    def set_frame_position(self, frame_no: int) -> None:
        """Jumps directly to a specific frame index in the video file."""
        if self._cap is None or not self._cap.isOpened():
            raise VideoSourceError("Replay engine is not initialized.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self._frame_index = frame_no

    def get_total_frames(self) -> int:
        """Returns the total number of frames in the video file."""
        if self._cap is None or not self._cap.isOpened():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def reset(self) -> None:
        """Resets the playback position back to the first frame."""
        self.set_frame_position(0)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReplayEngine":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
