import pytest
import numpy as np
import time
import threading
from src.ingestion.buffer import CircularFrameBuffer
from src.ingestion.sampler import FrameSampler
from src.ingestion.metrics import PerformanceMetricsTracker
from src.ingestion.interfaces import VideoSource

class MockStaticVideoSource(VideoSource):
    """Mock video source that generates fake frames for testing purposes."""

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480) -> None:
        self._fps = fps
        self._width = width
        self._height = height
        self._opened = False
        self._frame_count = 0

    def initialize(self) -> None:
        self._opened = True
        self._frame_count = 0
        
    def shutdown(self) -> None:
        self.release()

    def read(self):
        if not self._opened:
            return False, None
        
        # Create a blank mock frame
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_count += 1
        return True, frame

    def is_opened(self) -> bool:
        return self._opened

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self):
        return self._width, self._height

    def release(self) -> None:
        self._opened = False


def test_circular_frame_buffer_push_pop():
    buffer = CircularFrameBuffer(max_size=3)
    
    frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 1
    frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 2
    frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 3
    
    buffer.push(frame1, {"id": 1})
    buffer.push(frame2, {"id": 2})
    buffer.push(frame3, {"id": 3})
    
    assert buffer.size() == 3
    assert buffer.get_dropped_count() == 0
    
    # Push extra frame, evicting the oldest one (frame1)
    frame4 = np.ones((10, 10, 3), dtype=np.uint8) * 4
    buffer.push(frame4, {"id": 4})
    
    assert buffer.size() == 3
    assert buffer.get_dropped_count() == 1
    
    # Verify FIFO order
    f_pop1, meta1 = buffer.pop()
    assert meta1["id"] == 2  # frame1 got evicted, so frame2 is next
    assert np.all(f_pop1 == 2)
    
    f_pop2, meta2 = buffer.pop()
    assert meta2["id"] == 3
    
    f_pop3, meta3 = buffer.pop()
    assert meta3["id"] == 4
    
    assert buffer.size() == 0


def test_frame_sampler():
    # Downsample 30 FPS to 15 FPS (step = 2.0)
    sampler = FrameSampler(source_fps=30.0, target_fps=15.0)
    
    keeps = [sampler.should_keep() for _ in range(6)]
    # Expected: True, False, True, False, True, False (or similar alternating pattern)
    assert keeps == [True, False, True, False, True, False]


def test_performance_metrics_tracker():
    tracker = PerformanceMetricsTracker(window_size=10)
    
    # Record 5 frames with 10ms latency
    for _ in range(5):
        tracker.record_frame(10.0)
        time.sleep(0.01)  # small gap
        
    tracker.record_drop()
    summary = tracker.get_summary()
    
    assert summary["average_latency_ms"] == 10.0
    assert summary["total_processed"] == 5
    assert summary["total_dropped"] == 1
    assert summary["fps"] > 0.0
