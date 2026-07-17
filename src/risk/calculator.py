from typing import List, Dict
from src.risk.types import Evidence

class RiskScoreCalculator:
    """Computes shopper risk scores based on weighted evidence sums and temporal decay."""

    # Default weights for behavior categories
    DEFAULT_WEIGHTS = {
        "POCKET_CONCEALMENT": 0.8,
        "BAG_CONCEALMENT": 0.7,
        "RESTRICTED_AREA_INTRUSION": 0.5,
        "LOITERING": 0.2
    }

    def __init__(
        self,
        weights: Dict[str, float] = None,
        decay_rate_per_sec: float = 0.01
    ) -> None:
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._decay_rate_per_ms = decay_rate_per_sec / 1000.0

    def calculate_score(
        self,
        evidence_list: List[Evidence],
        current_timestamp_ms: float
    ) -> float:
        """Calculates risk score based on active evidence and decays over time."""
        if not evidence_list:
            return 0.0

        # 1. Sum up weighted active evidence confidences
        base_score = 0.0
        latest_event_time = 0.0

        for ev in evidence_list:
            weight = self._weights.get(ev.behavior_type, 0.2)
            base_score += weight * ev.confidence
            latest_event_time = max(latest_event_time, ev.timestamp_ms)

        # Cap the accumulated risk score at 1.0
        base_score = min(1.0, base_score)

        # 2. Apply temporal decay since the most recent suspicious event
        time_elapsed_ms = max(0.0, current_timestamp_ms - latest_event_time)
        decay = time_elapsed_ms * self._decay_rate_per_ms
        
        final_score = max(0.0, base_score - decay)
        return final_score
