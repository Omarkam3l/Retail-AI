"""FastAPI middleware for logging, timing, and error handling."""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger("APIMiddleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"{method} {path} -> {response.status_code} [{duration_ms:.1f}ms]"
            )
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"{method} {path} -> 500 [{duration_ms:.1f}ms] {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"}
            )
