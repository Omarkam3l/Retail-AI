import threading
from typing import Dict, Any, Optional

class ExecutionContext:
    """Carries localized states, frame indices, and metrics through a single pipeline run."""

    def __init__(self, camera_id: str, frame_index: int, timestamp_ms: float) -> None:
        self.camera_id = camera_id
        self.frame_index = frame_index
        self.timestamp_ms = timestamp_ms
        
        # State indicators
        self.metadata: Dict[str, Any] = {}
        self.profiling_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self.metadata[key] = value

    def get_meta(self, key: str) -> Optional[Any]:
        with self._lock:
            return self.metadata.get(key)

    def record_stage_latency(self, stage_name: str, duration_ms: float) -> None:
        """Records processing time for a specific stage of the cascade."""
        with self._lock:
            self.profiling_times[stage_name] = duration_ms
            
    def get_stage_latency(self, stage_name: str) -> float:
        with self._lock:
            return self.profiling_times.get(stage_name, 0.0)
            
    def get_all_latencies(self) -> Dict[str, float]:
        with self._lock:
            return dict(self.profiling_times)
