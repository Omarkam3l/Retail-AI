"""Pipeline and system metrics endpoints."""
import logging
from fastapi import APIRouter
from src.api.schemas import MetricsResponse
from src.api.dependencies import metrics_store

logger = logging.getLogger("MetricsRouter")
router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Returns current pipeline performance metrics."""
    data = metrics_store.to_dict()
    return MetricsResponse(**data)
