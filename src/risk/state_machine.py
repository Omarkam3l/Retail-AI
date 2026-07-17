from src.risk.types import RiskLevel

class RiskStateMachine:
    """Manages risk level transitions with hysteresis to prevent state oscillation."""

    def __init__(
        self,
        initial_level: RiskLevel = RiskLevel.LOW,
        escalate_to_med: float = 0.40,
        escalate_to_high: float = 0.75,
        deescalate_to_med: float = 0.60,
        deescalate_to_low: float = 0.30
    ) -> None:
        self._level = initial_level
        self._escalate_to_med = escalate_to_med
        self._escalate_to_high = escalate_to_high
        self._deescalate_to_med = deescalate_to_med
        self._deescalate_to_low = deescalate_to_low

    @property
    def level(self) -> RiskLevel:
        return self._level

    def update(self, score: float) -> tuple[RiskLevel, bool]:
        """Updates risk level based on the current score and hysteresis thresholds.

        Returns:
            A tuple of (updated_level, changed_flag)
        """
        old_level = self._level
        
        if self._level == RiskLevel.LOW:
            if score >= self._escalate_to_med:
                self._level = RiskLevel.MEDIUM
                if score >= self._escalate_to_high:
                    self._level = RiskLevel.HIGH
                    
        elif self._level == RiskLevel.MEDIUM:
            if score >= self._escalate_to_high:
                self._level = RiskLevel.HIGH
            elif score < self._deescalate_to_low:
                self._level = RiskLevel.LOW
                
        elif self._level == RiskLevel.HIGH:
            if score < self._deescalate_to_med:
                self._level = RiskLevel.MEDIUM
                if score < self._deescalate_to_low:
                    self._level = RiskLevel.LOW

        changed = (self._level != old_level)
        return self._level, changed
