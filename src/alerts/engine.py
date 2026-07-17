import logging
import uuid
from typing import List, Dict, Optional
import numpy as np

from src.alerts.interfaces import BaseAlertEngine
from src.alerts.types import Alert, AlertMetadata
from src.alerts.policy import AlertPolicyEngine
from src.alerts.cooldown import CooldownManager
from src.alerts.blur import FaceBlurProcessor
from src.alerts.evidence import EvidenceClipGenerator
from src.alerts.repository import AlertRepository
from src.alerts.dispatcher import BaseNotificationDispatcher, MockNotificationDispatcher
from src.alerts.audit import AuditLogger
from src.common.types import FrameMetadata, BoundingBox
from src.risk.types import RiskEvent

logger = logging.getLogger("AlertEvidenceEngine")

class AlertEvidenceEngine(BaseAlertEngine):
    """Production-grade coordinator for alerting, anonymization, and notification dispatching."""

    def __init__(
        self,
        policy_engine: Optional[AlertPolicyEngine] = None,
        cooldown_manager: Optional[CooldownManager] = None,
        blur_processor: Optional[FaceBlurProcessor] = None,
        clip_generator: Optional[EvidenceClipGenerator] = None,
        repository: Optional[AlertRepository] = None,
        dispatcher: Optional[BaseNotificationDispatcher] = None,
        audit_logger: Optional[AuditLogger] = None
    ) -> None:
        self._policy = policy_engine or AlertPolicyEngine()
        self._cooldown = cooldown_manager or CooldownManager()
        self._blur = blur_processor or FaceBlurProcessor()
        self._clip = clip_generator or EvidenceClipGenerator()
        self._repo = repository or AlertRepository()
        self._dispatch = dispatcher or MockNotificationDispatcher()
        self._audit = audit_logger or AuditLogger()
        
        # Internal cache of risk events to process in the current evaluate cycle
        self._ingested_risk_events: List[RiskEvent] = []

    def ingest_risk_events(self, events: List[RiskEvent]) -> None:
        """Caches risk events to be evaluated in the next processing frame loop."""
        self._ingested_risk_events.extend(events)

    def evaluate(
        self,
        risk_scores: Dict[int, float],
        frame_metadata: FrameMetadata
    ) -> List[Alert]:
        """Evaluates risk events against policies, checks cooldowns, and generates alerts."""
        alerts: List[Alert] = []
        events_to_process = list(self._ingested_risk_events)
        self._ingested_risk_events.clear()

        # In case no risk events were explicitly ingested, we can generate mock events for any 
        # HIGH risk scores to ensure backward compatibility and robust pipeline fallbacks.
        if not events_to_process:
            from src.risk.types import RiskLevel
            for tid, score in risk_scores.items():
                if score >= 75.0:
                    events_to_process.append(
                        RiskEvent(
                            track_id=tid,
                            risk_level=RiskLevel.HIGH,
                            previous_level=RiskLevel.LOW,
                            score=score,
                            timestamp_ms=frame_metadata.timestamp_ms,
                            evidence_list=[]
                        )
                    )

        for event in events_to_process:
            # 1. Match risk event to alerting policy
            alert_level = self._policy.evaluate_policy(event)
            if alert_level is None:
                continue

            event_type = f"SUSPICIOUS_{event.risk_level.value}"

            # 2. Check cooldown manager (prevent spamming alerts for same track ID)
            if self._cooldown.is_on_cooldown(event.track_id, event_type, event.timestamp_ms):
                logger.debug(f"Alert for track {event.track_id} is on cooldown.")
                continue

            # 3. Create clip metadata
            clip_meta = self._clip.generate_clip_metadata(
                event,
                frame_metadata.camera_id,
                frame_metadata.frame_index
            )

            # 4. Generate Alert incident
            alert_id = str(uuid.uuid4())
            alert = Alert(
                id=alert_id,
                track_id=event.track_id,
                camera_id=frame_metadata.camera_id,
                timestamp_ms=event.timestamp_ms,
                level=alert_level,
                event_type=event_type,
                clip_path=f"clips/{alert_id}.mp4",
                anonymized_image_path=f"images/{alert_id}_anonymized.jpg",
                metadata=clip_meta
            )

            # 5. Persist, Trigger Cooldown, Log, and Dispatch
            self._repo.save(alert)
            self._cooldown.trigger_alert(event.track_id, event_type, event.timestamp_ms)
            self._audit.log_alert_incident(alert)
            self._dispatch.dispatch(alert)
            
            alerts.append(alert)

        return alerts

    def anonymize_faces(
        self,
        frames: List[np.ndarray],
        face_boxes: List[List[BoundingBox]]
    ) -> List[np.ndarray]:
        """Loops through sequence of frames and face regions, applying Gaussian blur filter."""
        blurred_frames = []
        for i, frame in enumerate(frames):
            boxes = face_boxes[i] if i < len(face_boxes) else []
            blurred_frame = frame.copy()
            for box in boxes:
                blurred_frame = self._blur.anonymize_region(blurred_frame, box)
            blurred_frames.append(blurred_frame)
        return blurred_frames

    def compile_evidence_clip(
        self,
        frames: List[np.ndarray],
        output_path: str
    ) -> str:
        """Mock file writer saving frames into a loop output file."""
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Write dummy file to represent compiled video
        with open(output_path, "wb") as f:
            f.write(b"MOCK_VIDEO_CLIP_DATA")
        logger.info(f"Evidence clip compiled successfully at path: {output_path}")
        return output_path
