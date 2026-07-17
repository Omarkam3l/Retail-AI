from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from src.common.types import FrameMetadata, BoundingBox
from src.alerts.types import Alert

class BaseAlertEngine(ABC):
    """Abstract base class defining the contract for evaluation and compilation of store alerts."""

    @abstractmethod
    def evaluate(
        self,
        risk_scores: Dict[int, float],
        frame_metadata: FrameMetadata
    ) -> List[Alert]:
        """Evaluates risk scores against thresholds, enforcing lockout filters.

        Args:
            risk_scores: Current risk scores per active track.
            frame_metadata: Current frame attributes.

        Returns:
            A list of compiled Alert objects.
        """
        pass

    @abstractmethod
    def anonymize_faces(
        self,
        frames: List[np.ndarray],
        face_boxes: List[List[BoundingBox]]
    ) -> List[np.ndarray]:
        """Applies real-time Gaussian face blurring to frame lists for privacy compliance.

        Args:
            frames: Sequence of video frames.
            face_boxes: Nested list of face bounding boxes corresponding to each frame.

        Returns:
            A list of blurred (anonymized) frames.
        """
        pass

    @abstractmethod
    def compile_evidence_clip(
        self,
        frames: List[np.ndarray],
        output_path: str
    ) -> str:
        """Compresses a sequence of frames into a lightweight loop format (MP4/GIF).

        Args:
            frames: Sequence of blurred frames.
            output_path: Target disk path.

        Returns:
            The local file path to the compiled media clip.
        """
        pass
