from abc import ABC, abstractmethod
import logging
from src.alerts.types import Alert

logger = logging.getLogger("NotificationDispatcher")

class BaseNotificationDispatcher(ABC):
    """Abstract interface defining the contract for dispatching alert notifications."""

    @abstractmethod
    def dispatch(self, alert: Alert) -> None:
        """Sends alert notification payload to external endpoints (SMS, Slack, Emails)."""
        pass


class MockNotificationDispatcher(BaseNotificationDispatcher):
    """Concrete mock notification dispatcher logging triggers for dashboard feeds."""

    def __init__(self) -> None:
        self.dispatched_alerts: list[Alert] = []

    def dispatch(self, alert: Alert) -> None:
        self.dispatched_alerts.append(alert)
        logger.info(
            f"DISPATCH NOTIFICATION -> Camera: {alert.camera_id} | Track: {alert.track_id} | "
            f"Level: {alert.level.value} | Event: {alert.event_type}"
        )
