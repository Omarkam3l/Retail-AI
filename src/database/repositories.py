"""CRUD repositories for all database tables."""
import uuid
import logging
from typing import List, Optional, Dict, Any
from src.database.connection import DatabaseConnection
from src.database.models import AlertRecord, CameraRecord, EventRecord, SystemLogRecord, BenchmarkRecord

logger = logging.getLogger("DatabaseRepositories")


class AlertRepository:
    """CRUD operations for alerts table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def insert(self, alert: AlertRecord) -> str:
        self._db.execute(
            "INSERT INTO alerts (id, track_id, camera_id, timestamp_ms, level, event_type, clip_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alert.id, alert.track_id, alert.camera_id, alert.timestamp_ms, alert.level, alert.event_type, alert.clip_path)
        )
        return alert.id

    def get_by_id(self, alert_id: str) -> Optional[AlertRecord]:
        row = self._db.fetchone("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        if not row:
            return None
        return AlertRecord(**{k: row[k] for k in ["id", "track_id", "camera_id", "timestamp_ms", "level", "event_type", "clip_path", "created_at"]})

    def list_all(self, limit: int = 100, offset: int = 0, level: Optional[str] = None) -> List[AlertRecord]:
        query = "SELECT * FROM alerts"
        params: list = []
        if level:
            query += " WHERE level = ?"
            params.append(level)
        query += " ORDER BY timestamp_ms DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._db.fetchall(query, tuple(params))
        return [AlertRecord(**{k: r[k] for k in ["id", "track_id", "camera_id", "timestamp_ms", "level", "event_type", "clip_path", "created_at"]}) for r in rows]

    def count(self) -> int:
        result = self._db.fetchone("SELECT COUNT(*) as c FROM alerts")
        return result["c"] if result else 0


class CameraRepository:
    """CRUD operations for cameras table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, camera: CameraRecord) -> str:
        self._db.execute(
            "INSERT OR REPLACE INTO cameras (camera_id, source, status, fps, resolution) VALUES (?, ?, ?, ?, ?)",
            (camera.camera_id, camera.source, camera.status, camera.fps, camera.resolution)
        )
        return camera.camera_id

    def get_by_id(self, camera_id: str) -> Optional[CameraRecord]:
        row = self._db.fetchone("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        if not row:
            return None
        return CameraRecord(**{k: row[k] for k in ["camera_id", "source", "status", "fps", "resolution", "created_at"]})

    def list_all(self) -> List[CameraRecord]:
        rows = self._db.fetchall("SELECT * FROM cameras ORDER BY camera_id")
        return [CameraRecord(**{k: r[k] for k in ["camera_id", "source", "status", "fps", "resolution", "created_at"]}) for r in rows]


class EventRepository:
    """CRUD operations for events table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def insert(self, event: EventRecord) -> str:
        self._db.execute(
            "INSERT INTO events (id, camera_id, event_type, track_id, timestamp_ms, details) VALUES (?, ?, ?, ?, ?, ?)",
            (event.id, event.camera_id, event.event_type, event.track_id, event.timestamp_ms, event.details)
        )
        return event.id

    def list_by_camera(self, camera_id: str, limit: int = 100) -> List[EventRecord]:
        rows = self._db.fetchall(
            "SELECT * FROM events WHERE camera_id = ? ORDER BY timestamp_ms DESC LIMIT ?",
            (camera_id, limit)
        )
        return [EventRecord(**{k: r[k] for k in ["id", "camera_id", "event_type", "track_id", "timestamp_ms", "details", "created_at"]}) for r in rows]

    def count(self) -> int:
        result = self._db.fetchone("SELECT COUNT(*) as c FROM events")
        return result["c"] if result else 0


class SystemLogRepository:
    """CRUD operations for system_logs table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def insert(self, log: SystemLogRecord) -> str:
        self._db.execute(
            "INSERT INTO system_logs (id, level, module, message) VALUES (?, ?, ?, ?)",
            (log.id, log.level, log.module, log.message)
        )
        return log.id

    def list_recent(self, limit: int = 50) -> List[SystemLogRecord]:
        rows = self._db.fetchall("SELECT * FROM system_logs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [SystemLogRecord(**{k: r[k] for k in ["id", "level", "module", "message", "created_at"]}) for r in rows]


class BenchmarkRepository:
    """CRUD operations for benchmark_runs table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def insert(self, record: BenchmarkRecord) -> str:
        self._db.execute(
            "INSERT INTO benchmark_runs (id, model_version, dataset, detection_f1, tracking_mota, execution_time_s) VALUES (?, ?, ?, ?, ?, ?)",
            (record.id, record.model_version, record.dataset, record.detection_f1, record.tracking_mota, record.execution_time_s)
        )
        return record.id

    def list_all(self) -> List[BenchmarkRecord]:
        rows = self._db.fetchall("SELECT * FROM benchmark_runs ORDER BY created_at DESC")
        return [BenchmarkRecord(**{k: r[k] for k in ["id", "model_version", "dataset", "detection_f1", "tracking_mota", "execution_time_s", "created_at"]}) for r in rows]
