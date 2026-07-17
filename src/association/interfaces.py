from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from src.common.types import TrackedPerson, DetectedObject, AssociationState

class BaseAssociationEngine(ABC):
    """Abstract base class defining the contract for person-product-bag association logic."""

    @abstractmethod
    def associate(
        self,
        frame: np.ndarray,
        persons: List[TrackedPerson],
        objects: List[DetectedObject]
    ) -> Dict[int, Dict[int, AssociationState]]:
        """Maps relations between active customer tracks and detected products or bags.

        Args:
            frame: Raw video frame.
            persons: List of currently tracked customers.
            objects: List of currently tracked bags, baskets, and products.

        Returns:
            A dictionary mapping:
            person_track_id -> {object_track_id -> AssociationState}

        Raises:
            PipelineError: If association matching logic fails.
        """
        pass
