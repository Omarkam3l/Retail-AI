from abc import ABC, abstractmethod
from typing import List, Dict
from src.behavior.types import BehaviorFlag

class BaseRiskEngine(ABC):
    """Abstract base class defining the contract for evaluation of customer risk scores."""

    @abstractmethod
    def calculate_risk(
        self,
        track_id: int,
        behavior_flags: List[BehaviorFlag]
    ) -> float:
        """Calculates a normalized risk score [0, 100] for a given customer track.

        Args:
            track_id: Customer's unique tracking identifier.
            behavior_flags: List of active behavior flags for the track.

        Returns:
            A float risk value between 0.0 (no risk) and 100.0 (maximum risk).

        Raises:
            PipelineError: If risk aggregation calculations fail.
        """
        pass

    @abstractmethod
    def get_all_risk_scores(self) -> Dict[int, float]:
        """Retrieves current risk scores for all active customer tracks.

        Returns:
            A dictionary mapping: track_id -> risk_score
        """
        pass
