import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from src.config.validation import AppConfig, DetectorConfig
from src.common.health import HealthChecker
from src.inference.lifecycle_manager import ModelLifecycleManager

def test_pydantic_validation():
    # Valid configurations pass
    config_dict = {
        "detector": {"model_path": "yolo11s.pt", "device": "cpu", "confidence_threshold": 0.5},
        "tracker": {"track_threshold": 0.5, "match_threshold": 0.8, "track_buffer": 30},
        "association": {"proximity_threshold": 0.25, "persistence_threshold": 5, "lost_threshold": 30},
        "behavior": {"max_sequence_gap_seconds": 10.0, "loiter_threshold_seconds": 15.0}
    }
    
    app_config = AppConfig(**config_dict)
    assert app_config.detector.model_path == "yolo11s.pt"

    # Invalid threshold raises error
    config_dict["detector"]["confidence_threshold"] = 5.0 # out of [0.0, 1.0] bounds
    with pytest.raises(ValidationError):
        AppConfig(**config_dict)


def test_health_checker():
    hc = HealthChecker(max_memory_percent=99.0)
    
    # 1. Test liveness check
    live_info = hc.check_liveness()
    assert live_info["is_alive"] is True
    assert "memory_percent" in live_info

    # 2. Test readiness check
    hc.set_component_status("detector", False)
    assert hc.check_readiness()["is_ready"] is False
    
    hc.set_component_status("detector", True)
    hc.set_component_status("tracker", True)
    assert hc.check_readiness()["is_ready"] is True


@patch("src.inference.lifecycle_manager.YOLO")
def test_model_lifecycle_manager(mock_yolo_class):
    # Setup mock YOLO
    mock_instance_1 = MagicMock()
    mock_instance_2 = MagicMock()
    
    # Returns mock_instance_1 on first call, mock_instance_2 on second
    mock_yolo_class.side_effect = [mock_instance_1, mock_instance_2]

    lm = ModelLifecycleManager()
    
    # 1. Load model
    m1 = lm.load_model("yolo_model", "model_v1.pt")
    assert lm.get_model("yolo_model") == mock_instance_1
    
    # 2. Hot swap model reference
    lm.hot_reload_model("yolo_model", "model_v2.pt")
    assert lm.get_model("yolo_model") == mock_instance_2
    
    lm.clear()
