from typing import Dict
import os

class FeatureFlags:
    """Manages system feature switches at runtime via environment variables."""
    
    _flags: Dict[str, bool] = {
        "ENABLE_POSE_ESTIMATION": True,
        "ENABLE_FACE_BLURRING": True,
        "ENABLE_CLOUD_SYNC": True,
        "USE_INT8_QUANTIZATION": False,
        "DRY_RUN_ALERTS": False
    }

    @classmethod
    def is_enabled(cls, flag_name: str) -> bool:
        """Checks if a specific feature flag is active."""
        env_val = os.getenv(flag_name)
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return cls._flags.get(flag_name, False)
