import threading
import time
import logging
from typing import Optional
from src.ingestion.interfaces import VideoSource
from src.ingestion.buffer import CircularFrameBuffer
from src.ingestion.sampler import FrameSampler
from src.ingestion.metrics import PerformanceMetricsTracker
from src.ingestion.exceptions import VideoSourceError

logger = logging.getLogger("VideoDecoder")

class FrameDecoder:
    """Orchestrates stream ingestion and decoding on a dedicated background thread."""

    def __init__(
        self,
        source: VideoSource,
        buffer: CircularFrameBuffer,
        sampler: Optional[FrameSampler] = None,
        metrics: Optional[PerformanceMetricsTracker] = None,
        target_fps: float = 15.0
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._sampler = sampler
        self._metrics = metrics or PerformanceMetricsTracker()
        self._target_fps = target_fps
        self._target_interval = 1.0 / target_fps

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background decoding thread."""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._source.initialize()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("VideoDecoder background thread started.")

    def stop(self) -> None:
        """Signals the background thread to stop."""
        self._stop_event.set()
        logger.info("Stop signal sent to VideoDecoder thread.")

    def join(self, timeout: Optional[float] = None) -> None:
        """Waits for the background thread to exit."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
            self._source.release()

    def _run_loop(self) -> None:
        frame_index = 0
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            
            # Record latency of read operation
            try:
                success, frame = self._source.read()
            except Exception as e:
                logger.error(f"Error reading from video source: {e}")
                self._metrics.record_drop()
                time.sleep(self._target_interval)
                continue

            if not success or frame is None:
                # If stream disconnected, wait and retry
                self._metrics.record_drop()
                time.sleep(0.1)
                continue

            # Frame rate downsampling check
            if self._sampler is not None and not self._sampler.should_keep():
                # Skip frame
                continue

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            
            metadata = {
                "frame_index": frame_index,
                "timestamp_ms": time.time() * 1000.0,
                "latency_ms": latency_ms
            }
            
            self._buffer.push(frame, metadata)
            self._metrics.record_frame(latency_ms)
            frame_index += 1

            # Control loop rate to match target FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = self._target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
