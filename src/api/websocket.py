"""WebSocket connection manager for real-time event broadcasting."""
import json
import logging
import asyncio
from typing import List, Dict, Any, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("WebSocketManager")


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events to all clients."""

    def __init__(self) -> None:
        self._active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Removes a WebSocket connection."""
        async with self._lock:
            if websocket in self._active_connections:
                self._active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self._active_connections)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcasts an event to all connected clients."""
        if not self._active_connections:
            return

        message = json.dumps({"type": event_type, "data": data})
        disconnected: List[WebSocket] = []

        async with self._lock:
            for connection in self._active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.append(connection)

            for conn in disconnected:
                if conn in self._active_connections:
                    self._active_connections.remove(conn)

    async def send_personal(self, websocket: WebSocket, event_type: str, data: Dict[str, Any]) -> None:
        """Sends a message to a specific client."""
        message = json.dumps({"type": event_type, "data": data})
        try:
            await websocket.send_text(message)
        except Exception:
            await self.disconnect(websocket)

    @property
    def active_count(self) -> int:
        return len(self._active_connections)


# Global singleton
ws_manager = ConnectionManager()
