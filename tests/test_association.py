import pytest
import numpy as np
from src.association.matcher import calculate_iou, SpatialMatcher
from src.association.lifecycle import AssociationLifecycleTracker
from src.association.engine import ObjectAssociationEngine
from src.common.types import BoundingBox, TrackedPerson, DetectedObject, ClassLabel, AssociationState, EventType

def test_calculate_iou():
    box1 = BoundingBox(0.0, 0.0, 2.0, 2.0)
    box2 = BoundingBox(1.0, 1.0, 3.0, 3.0)
    
    iou = calculate_iou(box1, box2)
    # intersection = 1.0, union = 4.0 + 4.0 - 1.0 = 7.0
    assert pytest.approx(iou) == 1.0 / 7.0
    
    # Non-overlapping
    box3 = BoundingBox(5.0, 5.0, 6.0, 6.0)
    assert calculate_iou(box1, box3) == 0.0


def test_spatial_matcher_hungarian():
    matcher = SpatialMatcher(proximity_threshold=0.5)
    
    # Person 1 at (0.2, 0.2), Person 2 at (0.8, 0.8)
    persons = [
        TrackedPerson(track_id=1, bbox=BoundingBox(0.1, 0.1, 0.3, 0.3), confidence=0.9),
        TrackedPerson(track_id=2, bbox=BoundingBox(0.7, 0.7, 0.9, 0.9), confidence=0.9)
    ]
    
    # Object 1 at (0.22, 0.22) - closer to Person 1
    # Object 2 at (0.78, 0.78) - closer to Person 2
    objects = [
        DetectedObject(class_label=ClassLabel.SHELF_ITEM, bbox=BoundingBox(0.2, 0.2, 0.24, 0.24), confidence=0.8, track_id=10),
        DetectedObject(class_label=ClassLabel.SHELF_ITEM, bbox=BoundingBox(0.75, 0.75, 0.81, 0.81), confidence=0.8, track_id=20)
    ]
    
    matches = matcher.match(persons, objects)
    
    assert len(matches) == 2
    # Assert correct assignments
    assert (1, 10) in [(m[0], m[1]) for m in matches]
    assert (2, 20) in [(m[0], m[1]) for m in matches]


def test_association_lifecycle():
    tracker = AssociationLifecycleTracker(persistence_threshold=3, lost_threshold=5)
    
    # Frame 1: Match starts Candidate
    events = tracker.update_associations([(1, 10, 0.9)], timestamp_ms=100.0)
    assert len(events) == 0
    assert tracker.get_association_state(1, 10) == AssociationState.CANDIDATE
    
    # Frame 2: Match persists
    tracker.update_associations([(1, 10, 0.9)], timestamp_ms=200.0)
    assert tracker.get_association_state(1, 10) == AssociationState.CANDIDATE
    
    # Frame 3: Match persists -> CONFIRMED (PRODUCT_PICKED event emitted)
    events = tracker.update_associations([(1, 10, 0.9)], timestamp_ms=300.0)
    assert len(events) == 1
    assert events[0].event_type == EventType.PRODUCT_PICKED
    assert tracker.get_association_state(1, 10) == AssociationState.ASSOCIATED

    # Frame 4: Missing -> WEAK
    tracker.update_associations([], timestamp_ms=400.0)
    assert tracker.get_association_state(1, 10) == AssociationState.WEAK


def test_object_association_engine():
    engine = ObjectAssociationEngine(proximity_threshold=0.5, persistence_threshold=2)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    persons = [TrackedPerson(track_id=1, bbox=BoundingBox(0.1, 0.1, 0.3, 0.3), confidence=0.9)]
    objects = [DetectedObject(class_label=ClassLabel.SHELF_ITEM, bbox=BoundingBox(0.15, 0.15, 0.25, 0.25), confidence=0.8, track_id=10)]
    
    # Step 1: Candidate association
    assoc_map = engine.associate(frame, persons, objects)
    assert assoc_map[1][10] == AssociationState.CANDIDATE
    
    # Step 2: Confirmed pickup
    assoc_map = engine.associate(frame, persons, objects)
    assert assoc_map[1][10] == AssociationState.ASSOCIATED
    events = engine.get_events()
    assert len(events) == 1
    assert events[0].event_type == EventType.PRODUCT_PICKED
