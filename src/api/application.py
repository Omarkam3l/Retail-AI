"""FastAPI application factory with lifespan management."""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import health, cameras, alerts, inference, metrics
from src.api.websocket import ws_manager
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("APIApplication")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    logger.info("Retail AI Surveillance API starting up...")
    yield
    logger.info("Retail AI Surveillance API shutting down...")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="Retail AI Surveillance Platform",
        description="Production-grade AI surveillance system with real-time detection, tracking, and alert management.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register routers
    app.include_router(health.router)
    app.include_router(cameras.router)
    app.include_router(alerts.router)
    app.include_router(inference.router)
    app.include_router(metrics.router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Echo back or handle client messages
                await ws_manager.send_personal(websocket, "ack", {"received": data})
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)

    return app


# Application instance
app = create_app()
