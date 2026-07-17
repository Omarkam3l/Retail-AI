from abc import ABC, abstractmethod
from typing import List, Optional
from src.alerts.types import Alert

class BaseAlertRepository(ABC):
    """Abstract interface defining the contract for database persistence of alerts."""

    @abstractmethod
    def save_alert(self, alert: Alert) -> None:
        """Saves a new alert instance to storage.

        Args:
            alert: The Alert dataclass to persist.

        Raises:
            DataStorageError: If database write fails.
        """
        pass

    @abstractmethod
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieves a single alert by its UUID.

        Args:
            alert_id: UUIDv4 identifier string.

        Returns:
            The Alert object, or None if not found.
        """
        pass

    @abstractmethod
    def list_unsynced_alerts(self) -> List[Alert]:
        """Retrieves a list of cached alerts that have not yet synced to cloud.

        Returns:
            A list of cached Alerts.
        """
        pass

    @abstractmethod
    def mark_as_synced(self, alert_id: str, clip_url: str) -> None:
        """Updates the status of a local alert cache to marked as synced with cloud URL.

        Args:
            alert_id: UUIDv4 of the alert.
            clip_url: S3 storage public URL.
        """
        pass
