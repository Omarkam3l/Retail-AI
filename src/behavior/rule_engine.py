import logging
from typing import List
from src.behavior.interfaces import BehaviorRule
from src.behavior.types import BehaviorFlag
from src.association.types import AssociationEvent

logger = logging.getLogger("RuleEngine")

class RuleEngine:
    """Registry and executor for independent BehaviorRule plugins (SOLID Open-Closed)."""

    def __init__(self) -> None:
        self._rules: List[BehaviorRule] = []

    def register_rule(self, rule: BehaviorRule) -> None:
        """Registers a new behavior rule plugin."""
        self._rules.append(rule)
        logger.info(f"Registered behavior rule plugin: {rule.__class__.__name__}")

    def evaluate_all(
        self,
        track_id: int,
        history: List[AssociationEvent]
    ) -> List[BehaviorFlag]:
        """Runs all registered rules against the event history of a track ID."""
        all_flags: List[BehaviorFlag] = []
        for rule in self._rules:
            try:
                flags = rule.evaluate(track_id, history)
                all_flags.extend(flags)
            except Exception as e:
                logger.error(f"Error executing rule {rule.__class__.__name__}: {e}")
        return all_flags

    def clear_rules(self) -> None:
        self._rules.clear()
