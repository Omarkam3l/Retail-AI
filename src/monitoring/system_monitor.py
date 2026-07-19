"""Collects system metrics: CPU, GPU, VRAM, RAM, FPS, latency, dropped frames, queue sizes."""
import time
import logging
import threading
from typing import Dict, Any
import psutil

logger = logging.getLogger("SystemMonitor")


class SystemMonitor:
    """Collects and exposes system resource metrics."""

    def __init__(self) -> None:
        self._dropped_frames: int = 0
        self._queue_sizes: Dict[str, int] = {}
        self._pipeline_fps: float = 0.0
        self._pipeline_latency_ms: float = 0.0
        self._lock = threading.Lock()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Collects current system metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_available": False,
            "gpu_name": "",
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_percent": 0.0,
        }

        # GPU metrics
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics["gpu_available"] = True
                metrics["gpu_name"] = gpu.name
                metrics["vram_total_gb"] = round(gpu.memoryTotal / 1024, 2)
                metrics["vram_used_gb"] = round(gpu.memoryUsed / 1024, 2)
                metrics["vram_percent"] = round(gpu.memoryUtil * 100, 1)
        except (ImportError, Exception):
            pass

        return metrics

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Returns pipeline-specific metrics."""
        with self._lock:
            return {
                "fps": self._pipeline_fps,
                "latency_ms": self._pipeline_latency_ms,
                "dropped_frames": self._dropped_frames,
                "queue_sizes": dict(self._queue_sizes)
            }

    def update_pipeline(self, fps: float = 0.0, latency_ms: float = 0.0) -> None:
        with self._lock:
            self._pipeline_fps = fps
            self._pipeline_latency_ms = latency_ms

    def record_dropped_frame(self) -> None:
        with self._lock:
            self._dropped_frames += 1

    def update_queue_size(self, name: str, size: int) -> None:
        with self._lock:
            self._queue_sizes[name] = size

    def get_all_metrics(self) -> Dict[str, Any]:
        """Returns combined system and pipeline metrics."""
        result = self.get_system_metrics()
        result.update(self.get_pipeline_metrics())
        return result
