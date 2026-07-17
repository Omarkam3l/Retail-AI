import logging
import time
from typing import Dict, List, Optional
from src.risk.interfaces import BaseRiskEngine
from src.risk.types import RiskLevel, Evidence, RiskEvent
from src.risk.collector import EvidenceCollector
from src.risk.calculator import RiskScoreCalculator
from src.risk.state_machine import RiskStateMachine
from src.risk.suppression import SuppressionEngine
from src.behavior.types import BehaviorFlag

logger = logging.getLogger("RiskAssessmentEngine")

class RiskAssessmentEngine(BaseRiskEngine):
    """Production-grade Risk Engine aggregating behaviors, calculating scores, and triggering events."""

    def __init__(
        self,
        evidence_ttl_seconds: float = 60.0,
        decay_rate_per_sec: float = 0.02,
        suppression_engine: Optional[SuppressionEngine] = None
    ) -> None:
        self._collector = EvidenceCollector(evidence_ttl_seconds=evidence_ttl_seconds)
        self._calculator = RiskScoreCalculator(decay_rate_per_sec=decay_rate_per_sec)
        self._suppression = suppression_engine or SuppressionEngine()
        
        # Key: track_id -> RiskStateMachine
        self._state_machines: Dict[int, RiskStateMachine] = {}
        # Key: track_id -> current risk score (scaled 0.0 to 100.0)
        self._scores: Dict[int, float] = {}
        # Active risk events emitted during updates
        self.events: List[RiskEvent] = []

    def calculate_risk(
        self,
        track_id: int,
        behavior_flags: List[BehaviorFlag]
    ) -> float:
        """Accumulates incoming behavior flags, decays past scores, and returns scaled [0, 100] risk."""
        # Use current system time if none is provided
        import time
        timestamp_ms = time.time() * 1000.0

        # Find if flags contain timestamp_ms to override
        if behavior_flags:
            timestamp_ms = max(f.timestamp_ms for f in behavior_flags)

        # 1. Store flags as evidence
        for flag in behavior_flags:
            # Skip duplicate flags by checking if evidence is already recorded
            existing = self._collector.get_evidence(track_id)
            if any(e.raw_event == flag for e in existing):
                continue
                
            evidence = Evidence(
                behavior_type=flag.behavior_type,
                confidence=flag.confidence,
                timestamp_ms=flag.timestamp_ms,
                raw_event=flag
            )
            self._collector.add_evidence(track_id, evidence)

        # Prune older evidence
        self._collector.clean_expired(timestamp_ms)

        # 2. Compute risk score
        active_evidence = self._collector.get_evidence(track_id)
        score_fraction = self._calculator.calculate_score(active_evidence, timestamp_ms)
        score_100 = score_fraction * 100.0
        self._scores[track_id] = score_100

        # 3. Transition state machine
        if track_id not in self._state_machines:
            self._state_machines[track_id] = RiskStateMachine()
            
        sm = self._state_machines[track_id]
        old_level = sm.level
        new_level, changed = sm.update(score_fraction)

        # 4. Handle escalations and suppressions
        if changed:
            # Check if escalation is blocked by suppression rule
            is_escalating = (
                (old_level == RiskLevel.LOW and new_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)) or
                (old_level == RiskLevel.MEDIUM and new_level == RiskLevel.HIGH)
            )
            
            if is_escalating and self._suppression.is_suppressed(track_id, score_fraction):
                # Revert change
                sm._level = old_level
            else:
                # Emit RiskEvent
                risk_event = RiskEvent(
                    track_id=track_id,
                    risk_level=new_level,
                    previous_level=old_level,
                    score=score_100,
                    timestamp_ms=timestamp_ms,
                    evidence_list=active_evidence
                )
                self.events.append(risk_event)
                logger.info(f"Risk transition: Track {track_id} {old_level.value} -> {new_level.value} (score={score_100:.1f})")

        return score_100

    def get_all_risk_scores(self) -> Dict[int, float]:
        """Retrieves risk scores for all active shopper tracks."""
        return dict(self._scores)

    def get_events(self) -> List[RiskEvent]:
        """Retrieves and clears the generated risk events queue."""
        events_slice = list(self.events)
        self.events.clear()
        return events_slice

    def shutdown(self) -> None:
        self._collector.clear()
        self._state_machines.clear()
        self._scores.clear()
        self.events.clear()
