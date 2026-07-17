import logging
from typing import List, Optional, Dict
import numpy as np

from src.behavior.interfaces import BaseBehaviorEngine
from src.behavior.memory import TemporalMemory
from src.behavior.rule_engine import RuleEngine
from src.behavior.types import BehaviorFlag
from src.association.types import AssociationEvent
from src.common.types import FrameMetadata, EventType

logger = logging.getLogger("BehaviorEngine")

class BehaviorEngine(BaseBehaviorEngine):
    """Production-grade Behavior Analysis Engine coordinating sliding memory and rule evaluations."""

    def __init__(
        self,
        memory: Optional[TemporalMemory] = None,
        rule_engine: Optional[RuleEngine] = None
    ) -> None:
        self._memory = memory or TemporalMemory()
        self._rule_engine = rule_engine or RuleEngine()

    def analyze(
        self,
        frame_metadata: FrameMetadata,
        association_events: List[AssociationEvent]
    ) -> List[BehaviorFlag]:
        """Processes incoming events, updates sliding windows, and executes rules."""
        timestamp_ms = frame_metadata.timestamp_ms
        
        # 1. Ingest new events into sliding memory
        for event in association_events:
            self._memory.append_event(event.person_track_id, event)

        # 2. Run clean cycle on expired historical events
        self._memory.clean_expired(float(timestamp_ms))

        # 3. Evaluate rules on all active tracks
        all_alerts: List[BehaviorFlag] = []
        
        # Collect track IDs currently represented in frame
        active_track_ids = [p.track_id for p in frame_metadata.persons]
        
        for track_id in active_track_ids:
            history = self._memory.get_history(track_id)
            if history:
                alerts = self._rule_engine.evaluate_all(track_id, history)
                all_alerts.extend(alerts)

        return all_alerts

    def register_rule(self, rule) -> None:
        """Helper to expose rule registration publicly."""
        self._rule_engine.register_rule(rule)

    def shutdown(self) -> None:
        self._memory.clear()
        self._rule_engine.clear_rules()
