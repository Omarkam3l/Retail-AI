"""Renders intermediate frame visualizations for the debugger stage directories."""
import os
import cv2
import numpy as np
from src.cv_diagnostics.types import FrameRecord

class PipelineFrameDebugger:
    """Exports and overlays pipeline stages for step-by-step visual debugging."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    def save_stage_image(self, frame: np.ndarray, stage_name: str, frame_index: int) -> str:
        path = os.path.join(self._output_dir, "frame_debug", f"frame_{frame_index:04d}_stage_{stage_name}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frame)
        return path
