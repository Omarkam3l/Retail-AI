import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.inference.orchestrator import PipelineOrchestrator
from src.inference.event_bus import EventBus
from src.detection.yolo_detector import YOLO11Detector
from src.tracking.adapter import ByteTrackAdapter
from src.association.engine import ObjectAssociationEngine
from src.behavior.engine import BehaviorEngine
from src.behavior.rules import PocketConcealmentRule
from src.common.types import DetectedObject, BoundingBox, ClassLabel
from src.behavior.types import BehaviorFlag

@patch("src.detection.yolo_detector.YOLO")
def test_pipeline_integration_runs(mock_yolo_class):
    # 1. Setup detector mocks
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "person", 39: "bottle"}
    mock_yolo_class.return_value = mock_model_instance

    # Mock predictions to return a person and a product bottle close to them
    box_person = MagicMock()
    box_person.cls = [0]
    box_person.conf = [0.9]
    box_person.xyxy = [np.array([100, 100, 200, 350])]

    box_item = MagicMock()
    box_item.cls = [39]
    box_item.conf = [0.8]
    box_item.xyxy = [np.array([120, 120, 160, 200])]

    mock_result = MagicMock()
    mock_result.boxes = [box_person, box_item]
    mock_model_instance.predict.return_value = [mock_result]

    # Initialize components
    detector = YOLO11Detector(model_path="mock.pt", device="cpu")
    
    # 2. Setup tracking & association components (persistence_threshold=2, lost_threshold=5 for fast test)
    tracker = ByteTrackAdapter(track_threshold=0.25)
    association_engine = ObjectAssociationEngine(proximity_threshold=0.5, persistence_threshold=2, lost_threshold=5)
    
    # 3. Setup behavior engine with PocketConcealmentRule
    behavior_engine = BehaviorEngine()
    behavior_engine.register_rule(PocketConcealmentRule(max_sequence_gap_seconds=10.0))
    
    # 4. Initialize Orchestrator
    event_bus = EventBus()
    orchestrator = PipelineOrchestrator(
        camera_id="cam_test",
        detector=detector,
        tracker=tracker,
        association_engine=association_engine,
        behavior_engine=behavior_engine,
        event_bus=event_bus
    )
    orchestrator.initialize()

    # Create listener queue for event bus alerts
    alert_logs = []
    def alert_listener(flag: BehaviorFlag):
        alert_logs.append(flag)

    event_bus.subscribe("behavior_flag", alert_listener)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run 4 visibility frames to initialize ByteTrack (2 frames) + confirm association (2 frames)
    for i in range(4):
        orchestrator.process_frame(frame, frame_index=i, timestamp_ms=1000.0 * (i + 1))
    
    # Check that confirmed association now exists
    active_assoc = association_engine._tracker.get_active_associations()
    assert len(active_assoc) > 0  # Person and Product should be associated
    
    # Now, product disappears (mock detector returns only person, product is gone)
    mock_result.boxes = [box_person]
    
    # Run 10 frames to trigger lost_threshold (5 frames) and trigger EXPIRED
    for i in range(4, 15):
        orchestrator.process_frame(frame, frame_index=i, timestamp_ms=1000.0 * (i + 1))

    # Check that the pocket concealment alert was triggered and received by our listener!
    assert len(alert_logs) == 1
    assert alert_logs[0].behavior_type == "POCKET_CONCEALMENT"
    
    # Verify profiling summary contains records for all stages
    summary = orchestrator.profiler.get_summary()
    assert "detection" in summary
    assert "tracking" in summary
    assert "association" in summary
    assert "behavior" in summary

    orchestrator.shutdown()
