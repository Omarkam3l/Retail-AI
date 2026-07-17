from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAlertController(ABC):
    """Abstract interface defining the contract for API routing of alert notifications."""

    @abstractmethod
    def handle_ingest_alert(self, request_payload: Dict[str, Any], binary_file: bytes) -> Dict[str, Any]:
        """Ingests a new alert from an Edge gateway and triggers Cloud persistence.

        Args:
            request_payload: Metadata JSON mapping camera details.
            binary_file: Raw video loop bytes.

        Returns:
            A dictionary containing status confirmation and clip URL.
        """
        pass

    @abstractmethod
    def handle_submit_feedback(self, alert_id: str, feedback_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Logs user verification feedback for an alert.

        Args:
            alert_id: UUIDv4 of the target alert.
            feedback_payload: JSON containing feedback status ('True Positive', 'False Positive').

        Returns:
            A status confirmation dictionary.
        """
        pass
