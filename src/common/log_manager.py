"""Configures structured logging with separate rotating file handlers."""
import os
import logging
import json
from logging.handlers import RotatingFileHandler
from typing import Dict

LOG_DIR = "logs"


class JsonFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(log_dir: str = LOG_DIR) -> Dict[str, logging.Logger]:
    """Sets up separate rotating log files for different components."""
    os.makedirs(log_dir, exist_ok=True)

    formatter = JsonFormatter()

    log_configs = {
        "api": {"file": "api.log", "loggers": ["APIApplication", "APIMiddleware", "APISecurity", "HealthRouter", "CameraRouter", "AlertRouter", "InferenceRouter", "MetricsRouter"]},
        "pipeline": {"file": "pipeline.log", "loggers": ["PipelineOrchestrator", "YOLO11Detector", "ByteTrackAdapter", "ObjectAssociationEngine", "BehaviorEngine"]},
        "alerts": {"file": "alerts.log", "loggers": ["AlertEvidenceEngine", "RiskAssessmentEngine", "ClipManager", "SnapshotManager"]},
        "errors": {"file": "errors.log", "loggers": []},  # Catches all ERROR+ level
        "benchmark": {"file": "benchmark.log", "loggers": ["BenchmarkRunner", "ExperimentTracker", "DetectionMetrics", "TrackingMetrics"]},
    }

    loggers = {}

    for category, config in log_configs.items():
        handler = RotatingFileHandler(
            os.path.join(log_dir, config["file"]),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        handler.setFormatter(formatter)

        if category == "errors":
            handler.setLevel(logging.ERROR)
            root = logging.getLogger()
            root.addHandler(handler)
        else:
            handler.setLevel(logging.DEBUG)
            for logger_name in config["loggers"]:
                lg = logging.getLogger(logger_name)
                lg.addHandler(handler)
                lg.setLevel(logging.DEBUG)
                loggers[logger_name] = lg

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.getLogger().setLevel(logging.DEBUG)

    return loggers
