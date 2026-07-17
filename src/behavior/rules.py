from typing import List, Optional
from src.behavior.interfaces import BehaviorRule
from src.behavior.types import BehaviorFlag, PrimitiveEvent
from src.association.types import AssociationEvent
from src.common.types import EventType

class PocketConcealmentRule(BehaviorRule):
    """Detects concealment gestures where a picked item disappears within a short time window."""

    def __init__(self, max_sequence_gap_seconds: float = 10.0) -> None:
        self._max_gap_ms = max_sequence_gap_seconds * 1000.0
        self._flagged_object_ids = set()

    def evaluate(
        self,
        track_id: int,
        event_history: List[AssociationEvent]
    ) -> List[BehaviorFlag]:
        flags: List[BehaviorFlag] = []
        
        # Look for PRODUCT_PICKED events
        pick_events = [e for e in event_history if e.event_type == EventType.PRODUCT_PICKED]
        disappear_events = [e for e in event_history if e.event_type == EventType.PRODUCT_DISAPPEARED]

        for pick in pick_events:
            if pick.object_track_id in self._flagged_object_ids:
                continue

            # Find a matching disappearance that happened after the pick
            matching_disappear = next(
                (d for d in disappear_events 
                 if d.object_track_id == pick.object_track_id 
                 and d.timestamp_ms > pick.timestamp_ms 
                 and (d.timestamp_ms - pick.timestamp_ms) <= self._max_gap_ms),
                None
            )

            if matching_disappear:
                self._flagged_object_ids.add(pick.object_track_id)
                # Convert AssociationEvents to PrimitiveEvents for evidence list
                evidence = [
                    PrimitiveEvent(
                        event_type=EventType.PRODUCT_PICKED,
                        track_id=track_id,
                        timestamp_ms=pick.timestamp_ms,
                        bbox=pick.bbox,
                        confidence=pick.confidence
                    ),
                    PrimitiveEvent(
                        event_type=EventType.PRODUCT_DISAPPEARED,
                        track_id=track_id,
                        timestamp_ms=matching_disappear.timestamp_ms,
                        bbox=matching_disappear.bbox,
                        confidence=matching_disappear.confidence
                    )
                ]
                
                # Confidence fusion (product of confidences)
                combined_conf = pick.confidence * matching_disappear.confidence
                
                flags.append(
                    BehaviorFlag(
                        behavior_type="POCKET_CONCEALMENT",
                        track_id=track_id,
                        confidence=combined_conf,
                        timestamp_ms=matching_disappear.timestamp_ms,
                        evidence_events=evidence
                    )
                )

        return flags


class LoiteringRule(BehaviorRule):
    """Detects when a customer remains in a camera zone beyond a configured loitering duration."""

    def __init__(self, loiter_threshold_seconds: float = 15.0) -> None:
        self._loiter_threshold_ms = loiter_threshold_seconds * 1000.0

    def evaluate(
        self,
        track_id: int,
        event_history: List[AssociationEvent]
    ) -> List[BehaviorFlag]:
        if len(event_history) < 2:
            return []

        # Compare first and last recorded event timestamp
        start_time = event_history[0].timestamp_ms
        end_time = event_history[-1].timestamp_ms
        duration = end_time - start_time

        if duration >= self._loiter_threshold_ms:
            evidence = [
                PrimitiveEvent(
                    event_type=EventType.PERSON_STATIONARY,
                    track_id=track_id,
                    timestamp_ms=end_time,
                    confidence=1.0
                )
            ]
            return [
                BehaviorFlag(
                    behavior_type="LOITERING",
                    track_id=track_id,
                    confidence=0.8,
                    timestamp_ms=end_time,
                    evidence_events=evidence
                )
            ]
        return []
