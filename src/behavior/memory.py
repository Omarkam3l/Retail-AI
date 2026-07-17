from collections import deque
import threading
from typing import Dict, List, Tuple
from src.association.types import AssociationEvent

class TemporalMemory:
    """Thread-safe sliding history buffer storing association events per customer track."""

    def __init__(self, ttl_seconds: float = 20.0) -> None:
        self._ttl_ms = ttl_seconds * 1000.0
        # Key: track_id -> time-ordered deque of events
        self._histories: Dict[int, deque[AssociationEvent]] = {}
        self._lock = threading.Lock()

    def append_event(self, track_id: int, event: AssociationEvent) -> None:
        """Appends a new association event to the track's sliding history window."""
        with self._lock:
            if track_id not in self._histories:
                self._histories[track_id] = deque()
            self._histories[track_id].append(event)
            self._clean_track_history(track_id, event.timestamp_ms)

    def get_history(self, track_id: int) -> List[AssociationEvent]:
        """Retrieves a copy of the event history list for a specific track."""
        with self._lock:
            if track_id in self._histories:
                return list(self._histories[track_id])
            return []

    def clean_expired(self, current_timestamp_ms: float) -> None:
        """Triggers cleaning across all active track queues based on event TTL."""
        with self._lock:
            expired_ids = []
            for track_id in self._histories.keys():
                self._clean_track_history(track_id, current_timestamp_ms)
                if not self._histories[track_id]:
                    expired_ids.append(track_id)
            
            # Remove empty histories
            for track_id in expired_ids:
                del self._histories[track_id]

    def clear(self) -> None:
        with self._lock:
            self._histories.clear()

    def _clean_track_history(self, track_id: int, current_timestamp_ms: float) -> None:
        """Removes events older than the configured TTL threshold."""
        queue = self._histories[track_id]
        while queue and (current_timestamp_ms - queue[0].timestamp_ms) > self._ttl_ms:
            queue.popleft()
