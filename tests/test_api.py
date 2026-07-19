"""FastAPI endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from src.api.application import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_system_status(client):
    response = client.get("/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "memory_percent" in data


def test_list_cameras_empty(client):
    response = client.get("/cameras")
    assert response.status_code == 200


def test_register_camera(client):
    response = client.post("/camera/register", json={
        "camera_id": "test_cam",
        "source": "test.mp4",
        "confidence_threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_list_alerts(client):
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "total" in data


def test_get_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_fps" in data


def test_camera_start_not_found(client):
    response = client.post("/camera/start?camera_id=nonexistent")
    assert response.status_code == 404
