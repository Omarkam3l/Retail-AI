import threading
from typing import Dict, List, Optional
from src.alerts.types import Alert

class AlertRepository:
    """Thread-safe in-memory storage manager for active and historical alert incidents."""

    def __init__(self) -> None:
        self._alerts: Dict[str, Alert] = {}
        self._lock = threading.Lock()

    def save(self, alert: Alert) -> None:
        """Saves or updates an alert incident in the repository."""
        with self._lock:
            self._alerts[alert.id] = alert

    def get(self, alert_id: str) -> Optional[Alert]:
        """Retrieves an alert incident by its unique ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def list_all(self) -> List[Alert]:
        """Lists all alert incidents in chronological order."""
        with self._lock:
            return sorted(self._alerts.values(), key=lambda a: a.timestamp_ms)

    def clear(self) -> None:
        with self._lock:
            self._alerts.clear()
