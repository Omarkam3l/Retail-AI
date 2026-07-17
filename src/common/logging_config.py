import logging
import logging.config
import sys
from typing import Dict, Any

def configure_logging(level: str = "INFO") -> None:
    """Configures structured JSON and stream logging for the platform."""
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout,
                "level": level
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level
        }
    }
    logging.config.dictConfig(logging_config)
