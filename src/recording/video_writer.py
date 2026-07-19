"""Thread-safe OpenCV-based video writer."""
import os
import logging
import threading
from typing import Optional
import numpy as np

logger = logging.getLogger("VideoWriter")


class VideoWriter:
    """Thread-safe video writer using OpenCV."""

    def __init__(self, output_path: str, fps: float = 30.0,
                 width: int = 1920, height: int = 1080,
                 codec: str = "mp4v") -> None:
        self._output_path = output_path
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec
        self._writer = None
        self._lock = threading.Lock()
        self._frame_count = 0

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    def open(self) -> None:
        """Opens the video writer."""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (self._width, self._height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self._output_path}")
        logger.info(f"VideoWriter opened: {self._output_path} ({self._width}x{self._height} @ {self._fps}fps)")

    def write_frame(self, frame: np.ndarray) -> None:
        """Writes a single frame."""
        with self._lock:
            if self._writer is None:
                self.open()
            self._writer.write(frame)
            self._frame_count += 1

    def close(self) -> None:
        """Releases the video writer."""
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
                logger.info(f"VideoWriter closed: {self._output_path} ({self._frame_count} frames)")

    @property
    def frame_count(self) -> int:
        return self._frame_count
