import threading
from typing import Dict, List
from src.risk.types import Evidence

class EvidenceCollector:
    """Thread-safe evidence accumulator storing suspicious behaviors per customer track."""

    def __init__(self, evidence_ttl_seconds: float = 60.0) -> None:
        self._ttl_ms = evidence_ttl_seconds * 1000.0
        # Key: track_id -> List of Evidence
        self._evidence_map: Dict[int, List[Evidence]] = {}
        self._lock = threading.Lock()

    def add_evidence(self, track_id: int, evidence: Evidence) -> None:
        """Accumulates a new piece of behavior evidence for the shopper."""
        with self._lock:
            if track_id not in self._evidence_map:
                self._evidence_map[track_id] = []
            self._evidence_map[track_id].append(evidence)
            self._clean_expired_evidence_locked(track_id, evidence.timestamp_ms)

    def get_evidence(self, track_id: int) -> List[Evidence]:
        """Retrieves all active collected evidence for the shopper."""
        with self._lock:
            if track_id in self._evidence_map:
                return list(self._evidence_map[track_id])
            return []

    def clean_expired(self, current_timestamp_ms: float) -> None:
        """Prunes historical evidence that has aged past the configured TTL."""
        with self._lock:
            empty_tracks = []
            for track_id in self._evidence_map.keys():
                self._clean_expired_evidence_locked(track_id, current_timestamp_ms)
                if not self._evidence_map[track_id]:
                    empty_tracks.append(track_id)
            
            for track_id in empty_tracks:
                del self._evidence_map[track_id]

    def clear(self) -> None:
        with self._lock:
            self._evidence_map.clear()

    def _clean_expired_evidence_locked(self, track_id: int, current_timestamp_ms: float) -> None:
        """Internal helper to clean outdated evidence for a track."""
        history = self._evidence_map[track_id]
        self._evidence_map[track_id] = [
            ev for ev in history 
            if (current_timestamp_ms - ev.timestamp_ms) <= self._ttl_ms
        ]
