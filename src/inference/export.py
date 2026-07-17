import json
import logging
from typing import List, Any, Dict
from src.association.types import AssociationEvent
from src.behavior.types import BehaviorFlag

logger = logging.getLogger("TimelineExporter")

class JSONTimelineExporter:
    """Exports structured, chronological event logs from pipeline execution sessions."""

    @staticmethod
    def serialize_event(event: Any) -> Dict[str, Any]:
        """Serializes domain events to standard JSON-compatible dictionaries."""
        if isinstance(event, AssociationEvent):
            return {
                "event_category": "association",
                "event_type": str(event.event_type),
                "person_id": event.person_track_id,
                "object_id": event.object_track_id,
                "timestamp_ms": event.timestamp_ms,
                "confidence": float(event.confidence)
            }
        elif isinstance(event, BehaviorFlag):
            evidence = []
            for ev in event.evidence_events:
                evidence.append({
                    "event_type": str(ev.event_type),
                    "timestamp_ms": ev.timestamp_ms,
                    "confidence": float(ev.confidence)
                })
            return {
                "event_category": "behavior",
                "event_type": str(event.behavior_type),
                "person_id": event.track_id,
                "timestamp_ms": event.timestamp_ms,
                "confidence": float(event.confidence),
                "evidence": evidence
            }
        return {"type": type(event).__name__, "raw": str(event)}

    def export(self, events: List[Any], file_path: str) -> None:
        """Serializes and writes all events to a local JSON file."""
        logger.info(f"Exporting {len(events)} events to timeline log: {file_path}")
        serialized = [self.serialize_event(e) for e in events]
        
        try:
            with open(file_path, "w") as f:
                json.dump(serialized, f, indent=2)
            logger.info("Timeline log exported successfully.")
        except Exception as e:
            logger.error(f"Failed to export event timeline log: {e}")
