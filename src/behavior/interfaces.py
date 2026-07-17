from abc import ABC, abstractmethod
from typing import List, Dict
from src.common.types import FrameMetadata, AssociationState
from src.behavior.types import BehaviorFlag

class BaseBehaviorEngine(ABC):
    """Abstract base class defining the contract for analyzing customer actions."""

    @abstractmethod
    def analyze(
        self,
        frame_metadata: FrameMetadata,
        associations: Dict[int, Dict[int, AssociationState]]
    ) -> List[BehaviorFlag]:
        """Analyzes historical track states to detect suspicious actions.

        Args:
            frame_metadata: Current frame tracks and keypoints.
            associations: Active person-product-bag relations.

        Returns:
            A list of BehaviorFlag objects containing flagged actions.

        Raises:
            PipelineError: If event tracking or sequence evaluation fails.
        """
        pass
