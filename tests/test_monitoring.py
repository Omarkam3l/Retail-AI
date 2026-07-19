"""System monitor tests."""
import pytest
from src.monitoring.system_monitor import SystemMonitor


def test_system_metrics():
    monitor = SystemMonitor()
    metrics = monitor.get_system_metrics()
    assert "cpu_percent" in metrics
    assert "ram_total_gb" in metrics
    assert "gpu_available" in metrics


def test_pipeline_metrics():
    monitor = SystemMonitor()
    monitor.update_pipeline(fps=25.0, latency_ms=40.0)
    metrics = monitor.get_pipeline_metrics()
    assert metrics["fps"] == 25.0
    assert metrics["latency_ms"] == 40.0


def test_dropped_frames():
    monitor = SystemMonitor()
    monitor.record_dropped_frame()
    monitor.record_dropped_frame()
    metrics = monitor.get_pipeline_metrics()
    assert metrics["dropped_frames"] == 2


def test_queue_sizes():
    monitor = SystemMonitor()
    monitor.update_queue_size("detection", 5)
    monitor.update_queue_size("tracking", 3)
    metrics = monitor.get_pipeline_metrics()
    assert metrics["queue_sizes"]["detection"] == 5
    assert metrics["queue_sizes"]["tracking"] == 3


def test_all_metrics():
    monitor = SystemMonitor()
    all_metrics = monitor.get_all_metrics()
    assert "cpu_percent" in all_metrics
    assert "fps" in all_metrics
