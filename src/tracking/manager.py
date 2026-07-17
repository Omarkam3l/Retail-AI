import logging
from typing import Dict, List, Tuple, Optional
from src.common.types import BoundingBox
from src.tracking.state_machine import TrackState, TrackStateMachine

logger = logging.getLogger("TrackManager")

class TrackMetadata:
    """Stores history, state machine, and velocity details for an active track."""
    
    def __init__(self, track_id: int, bbox: BoundingBox, confidence: float) -> None:
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.state_machine = TrackStateMachine(TrackState.NEW)
        
        self.history: List[BoundingBox] = [bbox]
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.age_frames = 1
        self.missed_frames = 0

    def update(self, bbox: BoundingBox, confidence: float) -> None:
        """Updates bounding box history, age, and computes current velocity vector."""
        self.bbox = bbox
        self.confidence = confidence
        self.missed_frames = 0
        self.age_frames += 1
        
        # Calculate velocity based on displacement of centers
        old_center = self.history[-1].center
        new_center = bbox.center
        self.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
        
        self.history.append(bbox)
        if len(self.history) > 30:  # Keep last 30 frames
            self.history.pop(0)
            
        if self.state_machine.state == TrackState.NEW and self.age_frames >= 5:
            self.state_machine.transition_to(TrackState.CONFIRMED)
        elif self.state_machine.state in (TrackState.OCCLUDED, TrackState.LOST):
            self.state_machine.transition_to(TrackState.CONFIRMED)

    def mark_missing(self, max_occlusion_frames: int = 30, max_lost_frames: int = 90) -> None:
        """Increments missed frames count and handles state degradation."""
        self.missed_frames += 1
        
        if self.missed_frames > max_lost_frames:
            self.state_machine.transition_to(TrackState.EXPIRED)
        elif self.missed_frames > max_occlusion_frames:
            if self.state_machine.state != TrackState.LOST:
                self.state_machine.transition_to(TrackState.LOST)
        else:
            if self.state_machine.state != TrackState.OCCLUDED:
                self.state_machine.transition_to(TrackState.OCCLUDED)


class TrackManager:
    """Tracks identity trajectories, updating states and checking expiry."""

    def __init__(self, max_occlusion_frames: int = 30, max_lost_frames: int = 90) -> None:
        self._max_occlusion = max_occlusion_frames
        self._max_lost = max_lost_frames
        self._tracks: Dict[int, TrackMetadata] = {}

    def update(self, active_track_detections: List[Tuple[int, BoundingBox, float]]) -> None:
        """Updates existing tracks and creates new ones based on active frame matches.

        Args:
            active_track_detections: List of (track_id, BoundingBox, confidence)
        """
        active_ids = set()
        
        for track_id, bbox, confidence in active_track_detections:
            active_ids.add(track_id)
            if track_id in self._tracks:
                self._tracks[track_id].update(bbox, confidence)
            else:
                self._tracks[track_id] = TrackMetadata(track_id, bbox, confidence)
                logger.debug(f"New track created: ID {track_id}")

        # Degrade and clean missing tracks
        missing_ids = set(self._tracks.keys()) - active_ids
        for m_id in missing_ids:
            track = self._tracks[m_id]
            track.mark_missing(self._max_occlusion, self._max_lost)
            if track.state_machine.state == TrackState.EXPIRED:
                logger.debug(f"Track expired: ID {m_id}")
                del self._tracks[m_id]

    def get_track_metadata(self, track_id: int) -> Optional[TrackMetadata]:
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> Dict[int, TrackMetadata]:
        """Returns all non-expired tracks."""
        return {tid: t for tid, t in self._tracks.items() if t.state_machine.state != TrackState.EXPIRED}

    def clear(self) -> None:
        self._tracks.clear()
