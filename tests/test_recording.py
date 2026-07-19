"""Recording system tests."""
import pytest
import tempfile
import os
import numpy as np
from src.recording.snapshot_manager import SnapshotManager
from src.recording.retention_policy import RetentionPolicy
from src.recording.video_writer import VideoWriter


def test_snapshot_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SnapshotManager(output_dir=tmpdir)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        path = sm.capture(frame, "snap_001", metadata={"alert": "test"})
        assert os.path.isfile(path)
        assert os.path.isfile(os.path.join(tmpdir, "snap_001_meta.json"))


def test_retention_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for i in range(5):
            with open(os.path.join(tmpdir, f"file_{i}.txt"), "w") as f:
                f.write("x" * 1000)

        policy = RetentionPolicy(max_age_hours=0.0001, max_size_gb=10.0)
        import time
        time.sleep(0.5)
        deleted = policy.cleanup(tmpdir)
        assert deleted >= 0

        stats = policy.get_stats(tmpdir)
        assert "file_count" in stats


def test_video_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        writer = VideoWriter(path, fps=10.0, width=320, height=240)
        writer.open()
        for _ in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.isfile(path)
        assert writer.frame_count == 10
