import time
from typing import Tuple, Optional
import cv2
import numpy as np
from src.ingestion.interfaces import VideoSource
from src.ingestion.exceptions import VideoSourceError

class OpenCVVideoSource(VideoSource):
    """Base class for sources using OpenCV's VideoCapture."""

    def __init__(self, resource: str | int) -> None:
        self._resource = resource
        self._cap: Optional[cv2.VideoCapture] = None

    def initialize(self) -> None:
        if self._cap is not None:
            self.release()
        self._cap = cv2.VideoCapture(self._resource)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Failed to open video source: {self._resource}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        if self._cap is None or not self._cap.isOpened():
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap is None or not self._cap.isOpened():
            return 0, 0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def shutdown(self) -> None:
        self.release()


class FileVideoSource(OpenCVVideoSource):
    """Concrete video source implementation for files (e.g. MP4, AVI)."""
    
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)


class WebcamVideoSource(OpenCVVideoSource):
    """Concrete video source implementation for local webcams."""
    
    def __init__(self, device_index: int = 0) -> None:
        super().__init__(device_index)


class RTSPVideoSource(OpenCVVideoSource):
    """RTSP camera stream source incorporating automatic reconnection logic."""

    def __init__(self, rtsp_url: str, reconnect_timeout_seconds: float = 5.0) -> None:
        super().__init__(rtsp_url)
        self._reconnect_timeout = reconnect_timeout_seconds
        self._last_reconnect_time = 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        success, frame = super().read()
        if not success:
            self._attempt_reconnection()
            return False, None
        return True, frame

    def _attempt_reconnection(self) -> None:
        now = time.time()
        if now - self._last_reconnect_time >= self._reconnect_timeout:
            self._last_reconnect_time = now
            try:
                self.initialize()
            except VideoSourceError:
                # Silently catch; we will retry after the timeout expires
                pass
