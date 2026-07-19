"""Performance and accuracy metrics collector for evaluation."""
import time
import threading
from typing import Dict, Any


class PerformanceTracker:
    """Tracks latency, memory, throughput, and accuracy metrics."""

    def __init__(self) -> None:
        self._total_inferences = 0
        self._total_latencies_ms = 0.0
        self._unknown_detections = 0
        self._hits = 0
        self._lock = threading.Lock()

    def record_inference(self, latency_ms: float, is_unknown: bool, is_correct: bool = True) -> None:
        with self._lock:
            self._total_inferences += 1
            self._total_latencies_ms += latency_ms
            if is_unknown:
                self._unknown_detections += 1
            if is_correct:
                self._hits += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = self._total_latencies_ms / self._total_inferences if self._total_inferences > 0 else 0.0
            acc = self._hits / self._total_inferences if self._total_inferences > 0 else 0.0
            unknown_rate = self._unknown_detections / self._total_inferences if self._total_inferences > 0 else 0.0
            return {
                "total_inferences": self._total_inferences,
                "avg_latency_ms": avg_latency,
                "accuracy": acc,
                "unknown_rate": unknown_rate
            }
