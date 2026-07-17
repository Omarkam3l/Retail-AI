import os
import psutil
from typing import Dict, Any

class HealthChecker:
    """Evaluates liveness and readiness of surveillance pipeline components and resources."""

    def __init__(self, max_memory_percent: float = 90.0) -> None:
        self._max_memory_percent = max_memory_percent
        self._components_ready: Dict[str, bool] = {}

    def set_component_status(self, component_name: str, is_ready: bool) -> None:
        """Sets the readiness state of a specific pipeline component."""
        self._components_ready[component_name] = is_ready

    def check_liveness(self) -> Dict[str, Any]:
        """Checks resource thresholds (RAM, disk) to verify process viability."""
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        status = "OK"
        if memory_percent > self._max_memory_percent:
            status = "CRITICAL_MEMORY_USAGE"

        return {
            "status": status,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "is_alive": status == "OK"
        }

    def check_readiness(self) -> Dict[str, Any]:
        """Checks if all critical components (detector, tracker) are fully initialized."""
        all_ready = all(self._components_ready.values()) if self._components_ready else False
        
        status = "READY" if all_ready else "INITIALIZING"
        
        return {
            "status": status,
            "components": dict(self._components_ready),
            "is_ready": all_ready
        }
