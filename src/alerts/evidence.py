from typing import Dict, Any, Optional
from src.alerts.types import AlertMetadata
from src.risk.types import RiskEvent

class EvidenceClipGenerator:
    """Generates structured metadata and coordinates for exporting video evidence clips."""

    def __init__(self, pre_event_padding: int = 30, post_event_padding: int = 30) -> None:
        self._pre_padding = pre_event_padding
        self._post_padding = post_event_padding

    def generate_clip_metadata(
        self,
        risk_event: RiskEvent,
        camera_id: str,
        current_frame_index: int
    ) -> AlertMetadata:
        """Compiles clip ranges, timestamps, and padding requirements for evidence logs."""
        start_frame = max(0, current_frame_index - self._pre_padding)
        end_frame = current_frame_index + self._post_padding

        evidence_summary = (
            f"Risk state {risk_event.risk_level.value} triggered at frame {current_frame_index}. "
            f"Evidence: {len(risk_event.evidence_list)} suspicious behavior flags logged."
        )

        extra_details = {
            "camera_id": camera_id,
            "trigger_frame": current_frame_index,
            "clip_start_frame": start_frame,
            "clip_end_frame": end_frame,
            "associated_risk_score": risk_event.score
        }

        return AlertMetadata(
            pre_event_frames=self._pre_padding,
            post_event_frames=self._post_padding,
            raw_evidence_summary=evidence_summary,
            extra_details=extra_details
        )
