import pytest
import time
from src.risk.types import RiskLevel, Evidence, RiskEvent
from src.risk.state_machine import RiskStateMachine
from src.risk.calculator import RiskScoreCalculator
from src.risk.suppression import SuppressionEngine
from src.risk.engine import RiskAssessmentEngine
from src.behavior.types import BehaviorFlag

def test_risk_state_machine_hysteresis():
    # Setup state machine
    sm = RiskStateMachine(
        escalate_to_med=0.40,
        escalate_to_high=0.75,
        deescalate_to_med=0.60,
        deescalate_to_low=0.30
    )
    
    assert sm.level == RiskLevel.LOW
    
    # Update with score = 0.50 -> Escalates to MEDIUM
    lvl, changed = sm.update(0.50)
    assert lvl == RiskLevel.MEDIUM
    assert changed is True
    
    # Update with score = 0.70 -> Remains MEDIUM
    lvl, changed = sm.update(0.70)
    assert lvl == RiskLevel.MEDIUM
    assert changed is False
    
    # Update with score = 0.80 -> Escalates to HIGH
    lvl, changed = sm.update(0.80)
    assert lvl == RiskLevel.HIGH
    assert changed is True
    
    # Update with score = 0.65 -> Remains HIGH (hysteresis buffer requires score < 0.60 to fall to MEDIUM)
    lvl, changed = sm.update(0.65)
    assert lvl == RiskLevel.HIGH
    assert changed is False
    
    # Update with score = 0.55 -> De-escalates to MEDIUM
    lvl, changed = sm.update(0.55)
    assert lvl == RiskLevel.MEDIUM
    assert changed is True


def test_risk_score_calculator_decay():
    # Decay rate = 0.1 per second
    calc = RiskScoreCalculator(decay_rate_per_sec=0.1)
    
    evidence = [
        Evidence(behavior_type="POCKET_CONCEALMENT", confidence=1.0, timestamp_ms=1000.0, raw_event=None)
    ]
    
    # At t=1000ms: score should be weight of POCKET_CONCEALMENT (0.8)
    score1 = calc.calculate_score(evidence, current_timestamp_ms=1000.0)
    assert pytest.approx(score1) == 0.8
    
    # At t=3000ms (2 seconds later): score should decay by 2.0 * 0.1 = 0.2 -> score is 0.6
    score2 = calc.calculate_score(evidence, current_timestamp_ms=3000.0)
    assert pytest.approx(score2) == 0.6


def test_suppression_engine():
    se = SuppressionEngine()
    
    # Add rule to block shopper ID 99 from escalating
    se.register_suppression_rule(lambda tid, score: tid == 99)
    
    assert se.is_suppressed(99, 0.95) is True
    assert se.is_suppressed(1, 0.95) is False


def test_risk_assessment_engine_flow():
    engine = RiskAssessmentEngine(decay_rate_per_sec=0.0)
    
    flags = [
        BehaviorFlag(behavior_type="POCKET_CONCEALMENT", track_id=1, confidence=1.0, timestamp_ms=1000.0)
    ]
    
    # Calculate risk: should return 80.0 (0.8 weight * 100)
    score = engine.calculate_risk(track_id=1, behavior_flags=flags)
    assert pytest.approx(score) == 80.0
    
    # Fetch events: transition should be LOW -> HIGH (score 80 is >= 75)
    events = engine.get_events()
    assert len(events) == 1
    assert events[0].risk_level == RiskLevel.HIGH
    assert events[0].previous_level == RiskLevel.LOW
