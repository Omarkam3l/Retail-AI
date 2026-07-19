"""Dependency injection for FastAPI endpoints."""
import os
import threading
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger("APIDependencies")


class CameraRegistry:
    """Thread-safe registry of camera pipelines."""

    def __init__(self) -> None:
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(self, camera_id: str, source: str, **kwargs) -> None:
        with self._lock:
            self._cameras[camera_id] = {
                "source": source,
                "status": "inactive",
                "fps": 0.0,
                "frame_count": 0,
                "resolution": None,
                **kwargs
            }
        logger.info(f"Camera '{camera_id}' registered with source: {source}")

    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cameras.get(camera_id)

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._cameras)

    def update_status(self, camera_id: str, status: str) -> None:
        with self._lock:
            if camera_id in self._cameras:
                self._cameras[camera_id]["status"] = status

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)


class AlertStore:
    """In-memory alert store with pagination support."""

    def __init__(self) -> None:
        self._alerts: list = []
        self._lock = threading.Lock()

    def add(self, alert: Dict[str, Any]) -> None:
        with self._lock:
            self._alerts.append(alert)

    def get_page(self, page: int = 1, page_size: int = 50,
                 level: Optional[str] = None, camera_id: Optional[str] = None) -> tuple:
        with self._lock:
            filtered = self._alerts
            if level:
                filtered = [a for a in filtered if a.get("level") == level]
            if camera_id:
                filtered = [a for a in filtered if a.get("camera_id") == camera_id]

            total = len(filtered)
            start = (page - 1) * page_size
            end = start + page_size
            return filtered[start:end], total

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._alerts)


class MetricsStore:
    """Stores pipeline metrics."""

    def __init__(self) -> None:
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.total_alerts: int = 0
        self.pipeline_fps: float = 0.0
        self.avg_latency_ms: float = 0.0
        self.stage_latencies: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, fps: float = 0.0, latency: float = 0.0,
               detections: int = 0, stage_latencies: Dict[str, float] = None) -> None:
        with self._lock:
            self.total_frames += 1
            self.total_detections += detections
            self.pipeline_fps = fps
            self.avg_latency_ms = latency
            if stage_latencies:
                self.stage_latencies = stage_latencies

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pipeline_fps": self.pipeline_fps,
                "avg_latency_ms": self.avg_latency_ms,
                "total_frames_processed": self.total_frames,
                "total_detections": self.total_detections,
                "total_alerts": self.total_alerts,
                "stage_latencies": self.stage_latencies
            }


# Global singletons
camera_registry = CameraRegistry()
alert_store = AlertStore()
metrics_store = MetricsStore()
