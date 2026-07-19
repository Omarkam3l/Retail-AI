"""Health check and system status endpoints."""
import time
import logging
import psutil
from fastapi import APIRouter
from src.api.schemas import HealthResponse, SystemStatusResponse
from src.api.dependencies import camera_registry, alert_store

logger = logging.getLogger("HealthRouter")
router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns service health status."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        uptime_seconds=time.time() - _start_time
    )


@router.get("/system/status", response_model=SystemStatusResponse)
async def system_status() -> SystemStatusResponse:
    """Returns detailed system status including hardware metrics."""
    gpu_available = False
    gpu_memory_percent = 0.0
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_available = True
            gpu_memory_percent = gpus[0].memoryUtil * 100
    except (ImportError, Exception):
        pass

    return SystemStatusResponse(
        status="operational",
        active_cameras=len([c for c in camera_registry.list_all().values() if c["status"] == "active"]),
        total_alerts=alert_store.total_count,
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=psutil.virtual_memory().percent,
        gpu_available=gpu_available,
        gpu_memory_percent=gpu_memory_percent,
        uptime_seconds=time.time() - _start_time
    )
