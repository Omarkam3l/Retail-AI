import time
import collections
import threading
from typing import Dict, Any

class PerformanceMetricsTracker:
    """Thread-safe performance monitor logging FPS, latency, and drop statistics."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._frame_times = collections.deque(maxlen=window_size)
        self._latencies = collections.deque(maxlen=window_size)
        self._lock = threading.RLock()
        
        self._total_processed = 0
        self._total_dropped = 0

    def record_frame(self, latency_ms: float) -> None:
        """Logs a processed frame with its decoding latency."""
        with self._lock:
            now = time.time()
            self._frame_times.append(now)
            self._latencies.append(latency_ms)
            self._total_processed += 1

    def record_drop(self) -> None:
        """Logs a frame drop event."""
        with self._lock:
            self._total_dropped += 1

    def get_actual_fps(self) -> float:
        """Calculates the rolling average FPS over the configured window size."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            duration = self._frame_times[-1] - self._frame_times[0]
            if duration <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / duration

    def get_average_latency_ms(self) -> float:
        """Calculates the average decoding latency in milliseconds."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a snapshot dictionary of all performance metrics."""
        with self._lock:
            return {
                "fps": self.get_actual_fps(),
                "average_latency_ms": self.get_average_latency_ms(),
                "total_processed": self._total_processed,
                "total_dropped": self._total_dropped
            }
            
    def reset(self) -> None:
        """Resets all metrics counters."""
        with self._lock:
            self._frame_times.clear()
            self._latencies.clear()
            self._total_processed = 0
            self._total_dropped = 0
