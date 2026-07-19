"""API key validation and security utilities."""
import os
import logging
import hashlib
import time
from typing import Optional
from fastapi import Request, HTTPException, status

logger = logging.getLogger("APISecurity")

DEFAULT_API_KEY = "retail-ai-dev-key-2024"


def get_api_key() -> str:
    """Returns the configured API key from environment or default."""
    return os.environ.get("RETAIL_AI_API_KEY", DEFAULT_API_KEY)


async def verify_api_key(request: Request) -> bool:
    """Validates the API key from request headers.
    
    Skips validation for health and docs endpoints.
    """
    # Skip auth for health check and OpenAPI docs
    skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
    if request.url.path in skip_paths:
        return True

    api_key = request.headers.get("X-API-Key", "")
    expected = get_api_key()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header."
        )

    if not hashlib.sha256(api_key.encode()).hexdigest() == hashlib.sha256(expected.encode()).hexdigest():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )

    return True


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict = {}

    def check(self, client_ip: str) -> bool:
        now = time.time()
        # Clean old entries
        self._requests = {
            ip: timestamps
            for ip, timestamps in self._requests.items()
            if any(t > now - self._window for t in timestamps)
        }

        timestamps = self._requests.get(client_ip, [])
        timestamps = [t for t in timestamps if t > now - self._window]

        if len(timestamps) >= self._max_requests:
            return False

        timestamps.append(now)
        self._requests[client_ip] = timestamps
        return True
