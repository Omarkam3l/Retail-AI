import pytest
import numpy as np
from src.alerts.types import Alert, AlertLevel
from src.alerts.policy import AlertPolicyEngine
from src.alerts.cooldown import CooldownManager
from src.alerts.blur import FaceBlurProcessor
from src.alerts.evidence import EvidenceClipGenerator
from src.alerts.repository import AlertRepository
from src.alerts.dispatcher import MockNotificationDispatcher
from src.alerts.engine import AlertEvidenceEngine
from src.risk.types import RiskEvent, RiskLevel
from src.common.types import BoundingBox, FrameMetadata, TrackedPerson

def test_alert_policy_engine():
    pe = AlertPolicyEngine()
    
    # Risk MEDIUM -> AlertLevel MEDIUM
    event1 = RiskEvent(1, RiskLevel.MEDIUM, RiskLevel.LOW, 50.0, 1000.0, [])
    assert pe.evaluate_policy(event1) == AlertLevel.MEDIUM
    
    # Risk LOW -> None
    event2 = RiskEvent(1, RiskLevel.LOW, RiskLevel.LOW, 10.0, 1000.0, [])
    assert pe.evaluate_policy(event2) is None


def test_cooldown_manager():
    # 5-second cooldown
    cm = CooldownManager(cooldown_seconds=5.0)
    
    assert cm.is_on_cooldown(1, "SUSPICIOUS_HIGH", 1000.0) is False
    
    cm.trigger_alert(1, "SUSPICIOUS_HIGH", 1000.0)
    
    # At t=3000ms: within 5s cooldown window -> True
    assert cm.is_on_cooldown(1, "SUSPICIOUS_HIGH", 3000.0) is True
    
    # At t=7000ms: outside 5s cooldown window -> False
    assert cm.is_on_cooldown(1, "SUSPICIOUS_HIGH", 7000.0) is False


def test_face_blur_processor():
    bp = FaceBlurProcessor(kernel_size=5)
    
    # Use random noise so that Gaussian blur changes pixel values
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # BoundingBox at [0.2, 0.2, 0.5, 0.5] (pixel coordinate ROI [20:50, 20:50])
    bbox = BoundingBox(0.2, 0.2, 0.5, 0.5)
    
    blurred = bp.anonymize_region(frame, bbox)
    
    # Non-empty blurred region must differ from raw frame due to filtering
    assert not np.array_equal(frame[20:50, 20:50], blurred[20:50, 20:50])
    # Outside ROI must remain identical
    assert np.array_equal(frame[0:10, 0:10], blurred[0:10, 0:10])


def test_evidence_clip_generator():
    cg = EvidenceClipGenerator(pre_event_padding=10, post_event_padding=15)
    
    event = RiskEvent(1, RiskLevel.HIGH, RiskLevel.LOW, 85.0, 5000.0, [])
    
    meta = cg.generate_clip_metadata(event, camera_id="cam_01", current_frame_index=100)
    
    assert meta.pre_event_frames == 10
    assert meta.post_event_frames == 15
    assert meta.extra_details["clip_start_frame"] == 90
    assert meta.extra_details["clip_end_frame"] == 115


def test_alert_evidence_engine_pipeline():
    policy = AlertPolicyEngine()
    cooldown = CooldownManager()
    blur = FaceBlurProcessor()
    clip = EvidenceClipGenerator()
    repo = AlertRepository()
    dispatcher = MockNotificationDispatcher()
    
    engine = AlertEvidenceEngine(
        policy_engine=policy,
        cooldown_manager=cooldown,
        blur_processor=blur,
        clip_generator=clip,
        repository=repo,
        dispatcher=dispatcher
    )
    
    event = RiskEvent(1, RiskLevel.HIGH, RiskLevel.LOW, 90.0, 1000.0, [])
    engine.ingest_risk_events([event])
    
    person = TrackedPerson(1, BoundingBox(0.1, 0.1, 0.3, 0.3), 0.9)
    metadata = FrameMetadata(camera_id="cam_01", timestamp_ms=1000.0, frame_index=50, persons=[person])
    
    alerts = engine.evaluate(risk_scores={1: 90.0}, frame_metadata=metadata)
    
    assert len(alerts) == 1
    assert alerts[0].track_id == 1
    assert alerts[0].level == AlertLevel.HIGH
    
    # Assert saved to repository
    assert repo.get(alerts[0].id) == alerts[0]
    
    # Assert dispatched
    assert len(dispatcher.dispatched_alerts) == 1
    assert dispatcher.dispatched_alerts[0] == alerts[0]
