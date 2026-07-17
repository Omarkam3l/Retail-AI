import threading
from typing import Dict, Tuple

class CooldownManager:
    """Manages alert suppression windows per shopper/incident to prevent alert flooding."""

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self._cooldown_ms = cooldown_seconds * 1000.0
        # Key: (track_id, event_type) -> last_triggered_timestamp_ms
        self._last_triggered: Dict[Tuple[int, str], float] = {}
        self._lock = threading.Lock()

    def is_on_cooldown(self, track_id: int, event_type: str, timestamp_ms: float) -> bool:
        """Checks if a shopper is currently within the cooldown suppression window."""
        key = (track_id, event_type)
        with self._lock:
            if key in self._last_triggered:
                last_time = self._last_triggered[key]
                if (timestamp_ms - last_time) < self._cooldown_ms:
                    return True
            return False

    def trigger_alert(self, track_id: int, event_type: str, timestamp_ms: float) -> None:
        """Records the trigger timestamp, initiating a new cooldown suppression window."""
        key = (track_id, event_type)
        with self._lock:
            self._last_triggered[key] = timestamp_ms

    def clear(self) -> None:
        with self._lock:
            self._last_triggered.clear()
