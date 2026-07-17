import pytest
import numpy as np
from src.tracking.state_machine import TrackState, TrackStateMachine
from src.tracking.manager import TrackMetadata, TrackManager
from src.tracking.adapter import ByteTrackAdapter
from src.common.types import BoundingBox, DetectedObject, ClassLabel

def test_track_state_machine_transitions():
    sm = TrackStateMachine()
    assert sm.state == TrackState.NEW
    
    sm.transition_to(TrackState.CONFIRMED)
    assert sm.state == TrackState.CONFIRMED
    
    sm.transition_to(TrackState.OCCLUDED)
    assert sm.state == TrackState.OCCLUDED
    
    sm.transition_to(TrackState.LOST)
    assert sm.state == TrackState.LOST
    
    sm.transition_to(TrackState.EXPIRED)
    assert sm.state == TrackState.EXPIRED
    
    with pytest.raises(ValueError):
        # Cannot transition out of terminal state
        sm.transition_to(TrackState.CONFIRMED)


def test_track_metadata_velocity():
    bbox1 = BoundingBox(0.1, 0.1, 0.2, 0.2)
    bbox2 = BoundingBox(0.12, 0.15, 0.22, 0.25)
    
    meta = TrackMetadata(track_id=1, bbox=bbox1, confidence=0.9)
    assert meta.velocity == (0.0, 0.0)
    
    meta.update(bbox2, 0.95)
    
    # xc1 = 0.15, yc1 = 0.15
    # xc2 = 0.17, yc2 = 0.20
    # vx = 0.02, vy = 0.05
    assert pytest.approx(meta.velocity[0]) == 0.02
    assert pytest.approx(meta.velocity[1]) == 0.05
    assert meta.age_frames == 2


def test_track_manager_lifecycle():
    manager = TrackManager(max_occlusion_frames=2, max_lost_frames=4)
    
    bbox = BoundingBox(0.1, 0.1, 0.2, 0.2)
    
    # Frame 1: Create track
    manager.update([(1, bbox, 0.95)])
    assert 1 in manager.get_active_tracks()
    assert manager.get_track_metadata(1).state_machine.state == TrackState.NEW
    
    # Frame 2: Missing -> Degrades to OCCLUDED
    manager.update([])
    assert manager.get_track_metadata(1).state_machine.state == TrackState.OCCLUDED
    
    # Frame 3: Missing again
    manager.update([])
    assert manager.get_track_metadata(1).state_machine.state == TrackState.OCCLUDED
    
    # Frame 4: Missing -> Degrades to LOST
    manager.update([])
    assert manager.get_track_metadata(1).state_machine.state == TrackState.LOST
    
    # Frame 5: Re-detect -> Returns to CONFIRMED
    manager.update([(1, bbox, 0.9)])
    assert manager.get_track_metadata(1).state_machine.state == TrackState.CONFIRMED


def test_bytetrack_adapter_basic():
    # Construct adapter
    adapter = ByteTrackAdapter(track_threshold=0.25)
    adapter.initialize()
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Define a detected person
    detections = [
        DetectedObject(
            class_label=ClassLabel.PERSON,
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.3),
            confidence=0.85
        )
    ]
    
    persons, objects = adapter.track(frame, detections)
    
    # In the very first frame, ByteTrack might not instantly confirm the track, 
    # but the API execution itself must complete without errors.
    assert isinstance(persons, list)
    assert isinstance(objects, list)
