"""WebSocket connection manager tests."""
import pytest
from src.api.websocket import ConnectionManager


@pytest.mark.asyncio
async def test_connection_manager_init():
    manager = ConnectionManager()
    assert manager.active_count == 0
