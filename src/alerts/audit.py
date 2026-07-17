import logging
import json
from src.alerts.types import Alert

logger = logging.getLogger("AuditLogger")

class AuditLogger:
    """Writes immutable, structured logs of alert occurrences for security auditing."""

    def log_alert_incident(self, alert: Alert) -> None:
        """Serializes and writes an alert incident log record."""
        audit_record = {
            "action": "ALERT_GENERATED",
            "alert_id": alert.id,
            "camera_id": alert.camera_id,
            "track_id": alert.track_id,
            "timestamp_ms": alert.timestamp_ms,
            "level": alert.level.value,
            "event_type": alert.event_type,
            "evidence_summary": alert.metadata.raw_evidence_summary if alert.metadata else ""
        }
        
        # Write to structured log channel
        logger.info(f"AUDIT_RECORD: {json.dumps(audit_record)}")
