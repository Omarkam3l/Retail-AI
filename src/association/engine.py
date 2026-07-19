import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.association.interfaces import BaseAssociationEngine
from src.association.matcher import SpatialMatcher, calculate_iou
from src.association.lifecycle import AssociationLifecycleTracker
from src.association.types import AssociationEvent
from src.common.types import TrackedPerson, DetectedObject, AssociationState
from src.association.recovery import AssociationRecoveryEngine

logger = logging.getLogger("ObjectAssociationEngine")

class ObjectAssociationEngine(BaseAssociationEngine):
    """Main coordinator engine for object-person relational tracking."""

    def __init__(
        self,
        proximity_threshold: float = 0.25,
        persistence_threshold: int = 5,
        lost_threshold: int = 30,
        shelf_polygons: Optional[List[List[Tuple[float, float]]]] = None,
        recovery_threshold: float = 0.82,
        max_recovery_age_frames: int = 45,
        max_spatial_distance: float = 0.35
    ) -> None:
        self._matcher = SpatialMatcher(proximity_threshold=proximity_threshold)
        self._tracker = AssociationLifecycleTracker(
            persistence_threshold=persistence_threshold,
            lost_threshold=lost_threshold
        )
        self._shelf_polygons = shelf_polygons or []
        self._recovery_engine = AssociationRecoveryEngine(
            recovery_threshold=recovery_threshold,
            max_recovery_age_frames=max_recovery_age_frames,
            max_spatial_distance=max_spatial_distance
        )
        self._active_object_tracks: Dict[int, Tuple[DetectedObject, np.ndarray, float]] = {}
        self._frame_index = 0
        
        # Public events queue containing events generated in the last frame process
        self.events: List[AssociationEvent] = []
        self.mock_timestamp_ms: Optional[float] = None

    def associate(
        self,
        frame: np.ndarray,
        persons: List[TrackedPerson],
        objects: List[DetectedObject],
        object_embeddings: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, Dict[int, AssociationState]]:
        """Main coordinator pipeline matching entities, managing states, and returning records."""
        if self.mock_timestamp_ms is not None:
            timestamp_ms = self.mock_timestamp_ms
        else:
            import time
            timestamp_ms = time.time() * 1000.0
        
        self._frame_index += 1

        # 0. Track Recovery Logic
        if object_embeddings:
            # Detect inactive tracks from last frame and record them
            current_track_ids = {obj.track_id for obj in objects if obj.track_id is not None}
            for last_tid, (last_obj, last_emb, last_time) in list(self._active_object_tracks.items()):
                if last_tid not in current_track_ids:
                    self._recovery_engine.record_inactive(
                        track_id=last_tid,
                        class_label=last_obj.class_label.value if hasattr(last_obj.class_label, "value") else str(last_obj.class_label),
                        bbox=last_obj.bbox,
                        embedding=last_emb,
                        frame_index=self._frame_index - 1,
                        timestamp_ms=last_time
                    )
            
            # Attempt recovery for objects in the current frame
            recovered_objects = []
            for obj in objects:
                recovered_tid = None
                if obj.track_id is not None and obj.track_id in object_embeddings:
                    current_emb = object_embeddings[obj.track_id]
                    if obj.track_id not in self._active_object_tracks:
                        recovered_tid = self._recovery_engine.attempt_recovery(
                            obj,
                            current_emb,
                            frame_index=self._frame_index,
                            timestamp_ms=float(timestamp_ms)
                        )
                
                if recovered_tid is not None:
                    recovered_obj = DetectedObject(
                        class_label=obj.class_label,
                        bbox=obj.bbox,
                        confidence=obj.confidence,
                        track_id=recovered_tid,
                        sku=obj.sku,
                        brand=obj.brand,
                        category=obj.category,
                        similarity=obj.similarity,
                        rec_confidence=obj.rec_confidence
                    )
                    recovered_objects.append(recovered_obj)
                    if obj.track_id in object_embeddings:
                        object_embeddings[recovered_tid] = object_embeddings.pop(obj.track_id)
                else:
                    recovered_objects.append(obj)
            objects = recovered_objects

        # 1. Run Hungarian assignment to find active pairs
        matches = self._matcher.match(persons, objects, object_embeddings)
        
        # 2. Update lifecycle tracker and collect primitive events
        self.events = self._tracker.update_associations(matches, float(timestamp_ms))
        
        # 3. Shelf return verification
        # If a person was associated with a product, but that product overlaps a shelf polygon
        # and is far from the person, trigger a return event.
        self._verify_shelf_returns(persons, objects, float(timestamp_ms))
        
        # 4. Construct API return format
        return_dict: Dict[int, Dict[int, AssociationState]] = {}
        for (p_id, o_id), meta in self._tracker.get_active_associations().items():
            if p_id not in return_dict:
                return_dict[p_id] = {}
            return_dict[p_id][o_id] = meta.state

        # 5. Update active object tracks for the next frame
        self._active_object_tracks.clear()
        if object_embeddings:
            for obj in objects:
                if obj.track_id is not None and obj.track_id in object_embeddings:
                    self._active_object_tracks[obj.track_id] = (obj, object_embeddings[obj.track_id], float(timestamp_ms))

        return return_dict

    def get_events(self) -> List[AssociationEvent]:
        """Retrieves and clears the generated events queue."""
        events_slice = list(self.events)
        self.events.clear()
        return events_slice

    def shutdown(self) -> None:
        self._tracker.clear()
        self.events.clear()
        self._recovery_engine.clear()
        self._active_object_tracks.clear()
        self._frame_index = 0

    def _verify_shelf_returns(
        self,
        persons: List[TrackedPerson],
        objects: List[DetectedObject],
        timestamp_ms: float
    ) -> None:
        """Checks if associated products have been placed back on shelves."""
        active_associations = list(self._tracker.get_active_associations().keys())
        
        for p_id, o_id in active_associations:
            person = next((p for p in persons if p.track_id == p_id), None)
            obj = next((o for o in objects if o.track_id == o_id), None)
            
            if person is None or obj is None:
                continue
                
            # Check if object bounding box center is close to a shelf polygon or overlaps
            # In MVP, we can check if the product bbox is stationary and not overlapping the person
            iou = calculate_iou(person.bbox, obj.bbox)
            p_center = person.bbox.center
            o_center = obj.bbox.center
            dist = np.hypot(p_center[0] - o_center[0], p_center[1] - o_center[1])
            
            if iou == 0.0 and dist > 0.20:
                # If they are separated and product is inside a shelf region, force return
                # (For MVP, if we don't have explicit shelf coordinates, distance gap is sufficient)
                return_event = self._tracker.force_return(p_id, o_id, timestamp_ms)
                self.events.append(return_event)
                logger.info(f"Product return detected: Person {p_id} -> Product {o_id}")
