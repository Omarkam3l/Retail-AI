import logging
from typing import List, Callable

logger = logging.getLogger("SuppressionEngine")

class SuppressionEngine:
    """Evaluates override configurations to suppress false-positive risk alerts (e.g. employees)."""

    def __init__(self) -> None:
        # List of callables accepting (track_id, risk_score) -> returning bool (True if suppressed)
        self._rules: List[Callable[[int, float], bool]] = []

    def register_suppression_rule(self, rule: Callable[[int, float], bool]) -> None:
        """Registers a suppression rule callback."""
        self._rules.append(rule)
        logger.info(f"Registered suppression rule callback.")

    def is_suppressed(self, track_id: int, score: float) -> bool:
        """Evaluates registered suppression rules to check if escalation is blocked."""
        for rule in self._rules:
            try:
                if rule(track_id, score):
                    logger.debug(f"Risk escalation suppressed for track {track_id} by override rule.")
                    return True
            except Exception as e:
                logger.error(f"Error evaluating suppression override rule: {e}")
        return False

    def clear(self) -> None:
        self._rules.clear()
