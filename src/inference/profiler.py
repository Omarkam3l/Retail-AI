import collections
import threading
from typing import Dict, List, Any

class PipelineProfiler:
    """Logs and averages time spent executing each stage of the pipeline cascade."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._stage_latencies: Dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def record_latency(self, stage_name: str, duration_ms: float) -> None:
        """Records latency for a single run of a stage."""
        with self._lock:
            if stage_name not in self._stage_latencies:
                self._stage_latencies[stage_name] = collections.deque(maxlen=self._window_size)
            self._stage_latencies[stage_name].append(duration_ms)

    def get_average_latency(self, stage_name: str) -> float:
        """Calculates rolling average latency for a stage."""
        with self._lock:
            if stage_name not in self._stage_latencies or not self._stage_latencies[stage_name]:
                return 0.0
            queue = self._stage_latencies[stage_name]
            return sum(queue) / len(queue)

    def get_summary(self) -> Dict[str, float]:
        """Returns averages of all recorded stages."""
        with self._lock:
            return {
                stage: sum(q) / len(q) 
                for stage, q in self._stage_latencies.items() 
                if q
            }

    def reset(self) -> None:
        with self._lock:
            self._stage_latencies.clear()
