"""SQLite database tests."""
import pytest
import tempfile
import os
from src.database.connection import DatabaseConnection
from src.database.migrations import run_migrations
from src.database.repositories import AlertRepository, CameraRepository, EventRepository, BenchmarkRepository
from src.database.models import AlertRecord, CameraRecord, EventRecord, BenchmarkRecord


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = DatabaseConnection(db_path)
        run_migrations(conn)
        yield conn
        conn.close()


def test_migrations(db):
    result = db.fetchone("SELECT MAX(version) as v FROM schema_version")
    assert result["v"] >= 1


def test_alert_crud(db):
    repo = AlertRepository(db)
    alert = AlertRecord(id="alert-1", track_id=1, camera_id="cam1",
                        timestamp_ms=1000.0, level="HIGH", event_type="theft")
    repo.insert(alert)

    loaded = repo.get_by_id("alert-1")
    assert loaded is not None
    assert loaded.level == "HIGH"
    assert repo.count() == 1


def test_camera_crud(db):
    repo = CameraRepository(db)
    cam = CameraRecord(camera_id="cam1", source="rtsp://test", status="active")
    repo.upsert(cam)

    loaded = repo.get_by_id("cam1")
    assert loaded is not None
    assert loaded.status == "active"


def test_event_crud(db):
    repo = EventRepository(db)
    event = EventRecord(id="ev-1", camera_id="cam1", event_type="PERSON_ENTERED",
                        track_id=1, timestamp_ms=500.0)
    repo.insert(event)
    assert repo.count() == 1


def test_benchmark_crud(db):
    repo = BenchmarkRepository(db)
    record = BenchmarkRecord(id="bench-1", model_version="yolo11s",
                              dataset="test_v1", detection_f1=0.85)
    repo.insert(record)
    results = repo.list_all()
    assert len(results) == 1
    assert results[0].detection_f1 == 0.85
