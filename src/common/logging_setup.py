import logging
import json
import sys
from typing import Optional

class JSONFormatter(logging.Formatter):
    """Custom logging formatter serializing python logs into clean structured JSON formats."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt) or record.created,
            "level": record.levelname,
            "logger_name": record.name,
            "file": f"{record.pathname}:{record.lineno}",
            "message": record.getMessage()
        }
        
        # Include exception tracebacks if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


def setup_structured_logging(level: int = logging.INFO) -> None:
    """Configures the root logger to output structured JSON logs to standard stdout stream."""
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
