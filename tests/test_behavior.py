import pytest
import numpy as np
from src.behavior.memory import TemporalMemory
from src.behavior.rules import PocketConcealmentRule, LoiteringRule
from src.behavior.rule_engine import RuleEngine
from src.behavior.engine import BehaviorEngine
from src.association.types import AssociationEvent
from src.common.types import BoundingBox, FrameMetadata, TrackedPerson, EventType

def test_temporal_memory_ttl():
    # 2-second TTL
    memory = TemporalMemory(ttl_seconds=2.0)
    
    event1 = AssociationEvent(EventType.PRODUCT_PICKED, 1, 10, timestamp_ms=1000.0, confidence=0.9)
    event2 = AssociationEvent(EventType.PRODUCT_DISAPPEARED, 1, 10, timestamp_ms=2500.0, confidence=0.8)
    
    memory.append_event(1, event1)
    memory.append_event(1, event2)
    
    # At 2500ms, event1 (1000ms) has aged by 1500ms (within 2s TTL). Both should exist.
    assert len(memory.get_history(1)) == 2
    
    # At 3500ms, event1 (1000ms) has aged by 2500ms (exceeds 2s TTL). It should get cleaned.
    memory.clean_expired(current_timestamp_ms=3500.0)
    assert len(memory.get_history(1)) == 1
    assert memory.get_history(1)[0].timestamp_ms == 2500.0


def test_pocket_concealment_rule():
    rule = PocketConcealmentRule(max_sequence_gap_seconds=5.0)
    
    # Scenario: Pick item 10, then it disappears 2 seconds later
    history = [
        AssociationEvent(EventType.PRODUCT_PICKED, 1, 10, timestamp_ms=1000.0, confidence=0.9),
        AssociationEvent(EventType.PRODUCT_DISAPPEARED, 1, 10, timestamp_ms=3000.0, confidence=0.8)
    ]
    
    flags = rule.evaluate(track_id=1, event_history=history)
    
    assert len(flags) == 1
    assert flags[0].behavior_type == "POCKET_CONCEALMENT"
    # Combined confidence = 0.9 * 0.8 = 0.72
    assert pytest.approx(flags[0].confidence) == 0.72


def test_loitering_rule():
    # 10-second loitering threshold
    rule = LoiteringRule(loiter_threshold_seconds=10.0)
    
    # Customer present for 12 seconds
    history = [
        AssociationEvent(EventType.PRODUCT_PICKED, 1, 10, timestamp_ms=1000.0, confidence=1.0),
        AssociationEvent(EventType.PRODUCT_PICKED, 1, 10, timestamp_ms=13000.0, confidence=1.0)
    ]
    
    flags = rule.evaluate(track_id=1, event_history=history)
    assert len(flags) == 1
    assert flags[0].behavior_type == "LOITERING"


def test_behavior_engine_pipeline():
    engine = BehaviorEngine()
    engine.register_rule(PocketConcealmentRule())
    engine.register_rule(LoiteringRule())

    # Build metadata
    person = TrackedPerson(track_id=1, bbox=BoundingBox(0.1, 0.1, 0.3, 0.3), confidence=0.9)
    metadata = FrameMetadata(camera_id="cam_01", timestamp_ms=3000.0, frame_index=45, persons=[person])
    
    # Ingest PICK and DISAPPEAR events
    events = [
        AssociationEvent(EventType.PRODUCT_PICKED, 1, 10, timestamp_ms=1000.0, confidence=0.9),
        AssociationEvent(EventType.PRODUCT_DISAPPEARED, 1, 10, timestamp_ms=2500.0, confidence=0.8)
    ]
    
    alerts = engine.analyze(metadata, events)
    
    assert len(alerts) == 1
    assert alerts[0].behavior_type == "POCKET_CONCEALMENT"
