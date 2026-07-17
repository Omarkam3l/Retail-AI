from typing import Optional, Dict
from src.risk.types import RiskEvent, RiskLevel
from src.common.types import AlertLevel

class AlertPolicyEngine:
    """Evaluates RiskEvents against active policy rules to determine alert levels."""

    def __init__(self, risk_to_alert_mapping: Dict[RiskLevel, AlertLevel] = None) -> None:
        # Default mapping policy: Low is ignored, Medium/High map to corresponding alert levels
        self._mapping = risk_to_alert_mapping or {
            RiskLevel.MEDIUM: AlertLevel.MEDIUM,
            RiskLevel.HIGH: AlertLevel.HIGH
        }

    def evaluate_policy(self, risk_event: RiskEvent) -> Optional[AlertLevel]:
        """Evaluates a RiskEvent. Returns AlertLevel if it triggers an alert, else None."""
        return self._mapping.get(risk_event.risk_level)
