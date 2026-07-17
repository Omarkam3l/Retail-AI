import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.association.interfaces import BaseAssociationEngine
from src.association.matcher import SpatialMatcher, calculate_iou
from src.association.lifecycle import AssociationLifecycleTracker
from src.association.types import AssociationEvent
from src.common.types import TrackedPerson, DetectedObject, AssociationState

logger = logging.getLogger("ObjectAssociationEngine")

class ObjectAssociationEngine(BaseAssociationEngine):
    """Main coordinator engine for object-person relational tracking."""

    def __init__(
        self,
        proximity_threshold: float = 0.25,
        persistence_threshold: int = 5,
        lost_threshold: int = 30,
        shelf_polygons: Optional[List[List[Tuple[float, float]]]] = None
    ) -> None:
        self._matcher = SpatialMatcher(proximity_threshold=proximity_threshold)
        self._tracker = AssociationLifecycleTracker(
            persistence_threshold=persistence_threshold,
            lost_threshold=lost_threshold
        )
        self._shelf_polygons = shelf_polygons or []
        
        # Public events queue containing events generated in the last frame process
        self.events: List[AssociationEvent] = []
        self.mock_timestamp_ms: Optional[float] = None

    def associate(
        self,
        frame: np.ndarray,
        persons: List[TrackedPerson],
        objects: List[DetectedObject]
    ) -> Dict[int, Dict[int, AssociationState]]:
        """Main coordinator pipeline matching entities, managing states, and returning records."""
        if self.mock_timestamp_ms is not None:
            timestamp_ms = self.mock_timestamp_ms
        else:
            import time
            timestamp_ms = time.time() * 1000.0
        
        # 1. Run Hungarian assignment to find active pairs
        matches = self._matcher.match(persons, objects)
        
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

        return return_dict

    def get_events(self) -> List[AssociationEvent]:
        """Retrieves and clears the generated events queue."""
        events_slice = list(self.events)
        self.events.clear()
        return events_slice

    def shutdown(self) -> None:
        self._tracker.clear()
        self.events.clear()

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
