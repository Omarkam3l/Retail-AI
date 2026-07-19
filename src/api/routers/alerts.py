"""Alert query endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Query
from src.api.schemas import AlertResponse, AlertListResponse
from src.api.dependencies import alert_store

logger = logging.getLogger("AlertRouter")
router = APIRouter(tags=["Alerts"])


@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    level: Optional[str] = Query(default=None),
    camera_id: Optional[str] = Query(default=None)
) -> AlertListResponse:
    """Lists alerts with pagination and optional filtering."""
    alerts, total = alert_store.get_page(page, page_size, level, camera_id)
    return AlertListResponse(
        alerts=[
            AlertResponse(
                id=a.get("id", ""),
                track_id=a.get("track_id", 0),
                camera_id=a.get("camera_id", ""),
                timestamp_ms=a.get("timestamp_ms", 0.0),
                level=a.get("level", "LOW"),
                event_type=a.get("event_type", ""),
                clip_path=a.get("clip_path")
            )
            for a in alerts
        ],
        total=total,
        page=page,
        page_size=page_size
    )
