import time
from typing import Dict, List, Tuple
from src.common.types import AssociationState, EventType
from src.association.types import AssociationMetadata, AssociationEvent

class AssociationLifecycleTracker:
    """Tracks active associations, counts frames, and handles state degradation."""

    def __init__(self, persistence_threshold: int = 5, lost_threshold: int = 30) -> None:
        self._persistence_threshold = persistence_threshold
        self._lost_threshold = lost_threshold
        
        # Key: (person_id, object_id) -> AssociationMetadata
        self._associations: Dict[Tuple[int, int], AssociationMetadata] = {}

    def update_associations(
        self,
        active_matches: List[Tuple[int, int, float]],
        timestamp_ms: float
    ) -> List[AssociationEvent]:
        """Updates lifecycle states and generates pickup/returned/disappearance events.

        Args:
            active_matches: List of paired (person_id, object_id, confidence).
            timestamp_ms: Frame capture timestamp.

        Returns:
            A list of generated AssociationEvent objects.
        """
        events: List[AssociationEvent] = []
        active_pairs = set()

        # Update active matches
        for p_id, o_id, conf in active_matches:
            pair = (p_id, o_id)
            active_pairs.add(pair)
            
            if pair in self._associations:
                meta = self._associations[pair]
                meta.last_update_ms = timestamp_ms
                meta.missed_count = 0
                meta.confidence = conf
                
                if meta.state == AssociationState.CANDIDATE:
                    meta.persistence_count += 1
                    if meta.persistence_count >= self._persistence_threshold:
                        meta.state = AssociationState.ASSOCIATED
                        events.append(
                            AssociationEvent(
                                event_type=EventType.PRODUCT_PICKED,
                                person_track_id=p_id,
                                object_track_id=o_id,
                                timestamp_ms=timestamp_ms,
                                confidence=conf
                            )
                        )
                elif meta.state in (AssociationState.WEAK, AssociationState.LOST):
                    meta.state = AssociationState.ASSOCIATED
            else:
                # Initialize new Candidate association
                self._associations[pair] = AssociationMetadata(
                    person_track_id=p_id,
                    object_track_id=o_id,
                    state=AssociationState.CANDIDATE,
                    confidence=conf,
                    start_time_ms=timestamp_ms,
                    last_update_ms=timestamp_ms
                )

        # Handle missing associations
        missing_pairs = set(self._associations.keys()) - active_pairs
        expired_pairs = []

        for pair in missing_pairs:
            p_id, o_id = pair
            meta = self._associations[pair]
            meta.missed_count += 1
            
            if meta.state == AssociationState.CANDIDATE:
                # Candidate never confirmed, evict immediately
                expired_pairs.append(pair)
            elif meta.state == AssociationState.ASSOCIATED:
                meta.state = AssociationState.WEAK
            elif meta.state == AssociationState.WEAK:
                if meta.missed_count >= 5:
                    meta.state = AssociationState.LOST
            elif meta.state == AssociationState.LOST:
                if meta.missed_count >= self._lost_threshold:
                    meta.state = AssociationState.EXPIRED
                    expired_pairs.append(pair)
                    events.append(
                        AssociationEvent(
                            event_type=EventType.PRODUCT_DISAPPEARED,
                            person_track_id=p_id,
                            object_track_id=o_id,
                            timestamp_ms=timestamp_ms,
                            confidence=meta.confidence
                        )
                    )

        # Purge expired associations from active tracking dictionary
        for pair in expired_pairs:
            del self._associations[pair]

        return events

    def get_association_state(self, person_id: int, object_id: int) -> AssociationState:
        pair = (person_id, object_id)
        if pair in self._associations:
            return self._associations[pair].state
        return AssociationState.UNASSOCIATED

    def get_active_associations(self) -> Dict[Tuple[int, int], AssociationMetadata]:
        return self._associations

    def force_return(self, person_id: int, object_id: int, timestamp_ms: float) -> AssociationEvent:
        """Forces dissolution of an association, emitting a PRODUCT_RETURNED event."""
        pair = (person_id, object_id)
        confidence = 1.0
        if pair in self._associations:
            confidence = self._associations[pair].confidence
            del self._associations[pair]
            
        return AssociationEvent(
            event_type=EventType.PRODUCT_RETURNED,
            person_track_id=person_id,
            object_track_id=object_id,
            timestamp_ms=timestamp_ms,
            confidence=confidence
        )

    def clear(self) -> None:
        self._associations.clear()
