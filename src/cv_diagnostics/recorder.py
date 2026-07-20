"""Recorder layer serving as a single source of truth for observed frame metrics."""
from typing import Dict, List, Optional
from src.cv_diagnostics.types import FrameRecord

class DiagnosticsRecorder:
    """Stores sequential frame records for diagnostic access and report compiler generation."""

    def __init__(self) -> None:
        # Chronological index of recorded frames
        self._records: List[FrameRecord] = []

    def record_frame(self, record: FrameRecord) -> None:
        """Stores a frame-specific record."""
        self._records.append(record)

    def get_all_records(self) -> List[FrameRecord]:
        """Returns all recorded frame logs."""
        return self._records

    def get_record(self, frame_index: int) -> Optional[FrameRecord]:
        """Returns specific frame record if it exists."""
        for rec in self._records:
            if rec.frame_index == frame_index:
                return rec
        return None

    def clear(self) -> None:
        """Resets the recorder memory."""
        self._records.clear()
