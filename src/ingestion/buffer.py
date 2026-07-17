from collections import deque
from typing import Tuple, List, Optional
import threading
import numpy as np

class CircularFrameBuffer:
    """Thread-safe circular ring buffer for real-time video frame storage."""

    def __init__(self, max_size: int = 150) -> None:
        self._max_size = max_size
        self._buffer: deque[Tuple[np.ndarray, dict]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._dropped_frames_count = 0

    def push(self, frame: np.ndarray, metadata: dict) -> None:
        """Pushes a new frame and metadata to the buffer.
        
        If the buffer is full, the oldest frame is automatically evicted 
        to maintain real-time queue constraints.
        """
        with self._lock:
            if len(self._buffer) >= self._max_size:
                self._dropped_frames_count += 1
            self._buffer.append((frame, metadata))
            self._condition.notify_all()

    def pop(self, timeout: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """Pops the oldest frame and metadata from the buffer.
        
        Blocks the calling thread if the buffer is empty, up to the optional timeout.
        """
        with self._condition:
            while not self._buffer:
                success = self._condition.wait(timeout=timeout)
                if not success and not self._buffer:
                    raise IndexError("Pop timed out on empty circular buffer")
            return self._buffer.popleft()

    def get_all_frames(self) -> List[Tuple[np.ndarray, dict]]:
        """Retrieves a snapshot list of all frames currently stored in the buffer."""
        with self._lock:
            return list(self._buffer)

    def size(self) -> int:
        """Returns the current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    def max_size(self) -> int:
        """Returns the maximum capacity of the buffer."""
        return self._max_size

    def get_dropped_count(self) -> int:
        """Retrieves the total number of frames dropped due to buffer capacity limit."""
        with self._lock:
            return self._dropped_frames_count

    def clear(self) -> None:
        """Clears the buffer and resets drop metrics."""
        with self._lock:
            self._buffer.clear()
            self._dropped_frames_count = 0
