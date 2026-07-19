"""HTTP client wrapping all API calls for the dashboard."""
import os
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("DashboardAPIClient")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("RETAIL_AI_API_KEY", "retail-ai-dev-key-2024")


class APIClient:
    """Client for communicating with the Retail AI API."""

    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}

    def _get(self, path: str, params: Dict = None) -> Dict:
        try:
            resp = requests.get(f"{self._base_url}{path}", headers=self._headers, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": "API not reachable", "status": "offline"}
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str, json_data: Dict = None) -> Dict:
        try:
            resp = requests.post(f"{self._base_url}{path}", headers=self._headers, json=json_data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": "API not reachable"}
        except Exception as e:
            return {"error": str(e)}

    def health(self) -> Dict:
        return self._get("/health")

    def system_status(self) -> Dict:
        return self._get("/system/status")

    def list_cameras(self) -> List[Dict]:
        result = self._get("/cameras")
        if isinstance(result, list):
            return result
        return result.get("cameras", []) if isinstance(result, dict) else []

    def register_camera(self, camera_id: str, source: str, confidence: float = 0.35) -> Dict:
        return self._post("/camera/register", {
            "camera_id": camera_id, "source": source, "confidence_threshold": confidence
        })

    def start_camera(self, camera_id: str) -> Dict:
        return self._post(f"/camera/start?camera_id={camera_id}")

    def stop_camera(self, camera_id: str) -> Dict:
        return self._post(f"/camera/stop?camera_id={camera_id}")

    def list_alerts(self, page: int = 1, page_size: int = 50, level: str = None) -> Dict:
        params = {"page": page, "page_size": page_size}
        if level:
            params["level"] = level
        return self._get("/alerts", params)

    def get_metrics(self) -> Dict:
        return self._get("/metrics")
