"""Manages saving alert video clips with pre/post event windows."""
import os
import logging
import threading
import json
from collections import deque
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger("ClipManager")


class ClipManager:
    """Saves alert clips with configurable pre-event and post-event frame windows."""

    def __init__(self, output_dir: str = "data/clips",
                 pre_event_frames: int = 90,
                 post_event_frames: int = 60,
                 fps: float = 30.0) -> None:
        self._output_dir = output_dir
        self._pre_event_frames = pre_event_frames
        self._post_event_frames = post_event_frames
        self._fps = fps
        self._frame_buffer: deque = deque(maxlen=pre_event_frames)
        self._lock = threading.Lock()
        self._recording = False
        self._post_count = 0
        self._current_writer = None
        self._current_clip_id = ""
        os.makedirs(output_dir, exist_ok=True)

    def buffer_frame(self, frame: np.ndarray) -> None:
        """Adds a frame to the rolling buffer."""
        with self._lock:
            self._frame_buffer.append(frame.copy())

            if self._recording and self._current_writer is not None:
                self._current_writer.write_frame(frame)
                self._post_count += 1
                if self._post_count >= self._post_event_frames:
                    self._finalize_clip()

    def trigger_clip(self, alert_id: str, metadata: Dict[str, Any] = None) -> str:
        """Triggers clip saving starting from the buffer."""
        from src.recording.video_writer import VideoWriter

        with self._lock:
            clip_path = os.path.join(self._output_dir, f"{alert_id}.mp4")
            h, w = 480, 640
            if self._frame_buffer:
                h, w = self._frame_buffer[0].shape[:2]

            self._current_writer = VideoWriter(clip_path, fps=self._fps, width=w, height=h)
            self._current_writer.open()

            # Write buffered pre-event frames
            for frame in self._frame_buffer:
                self._current_writer.write_frame(frame)

            self._recording = True
            self._post_count = 0
            self._current_clip_id = alert_id

            # Save metadata
            if metadata:
                meta_path = os.path.join(self._output_dir, f"{alert_id}_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Clip triggered for alert '{alert_id}': {clip_path}")
            return clip_path

    def _finalize_clip(self) -> None:
        """Finalizes the current clip recording."""
        if self._current_writer:
            self._current_writer.close()
            self._current_writer = None
        self._recording = False
        logger.info(f"Clip finalized for alert '{self._current_clip_id}'.")
