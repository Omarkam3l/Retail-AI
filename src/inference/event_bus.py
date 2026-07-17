import logging
import threading
from typing import Dict, List, Callable, Any

logger = logging.getLogger("EventBus")

class EventBus:
    """Thread-safe publish-subscribe broker for clean decoupling of pipeline events."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Subscribes a callback listener function to an event channel."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscriber added for event type: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Removes a registered callback from an event channel."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Subscriber removed from event type: {event_type}")
                except ValueError:
                    pass

    def publish(self, event_type: str, event_data: Any) -> None:
        """Dispatches an event payload to all registered channel subscribers."""
        callbacks = []
        with self._lock:
            if event_type in self._subscribers:
                # Copy list to execute callbacks outside lock
                callbacks = list(self._subscribers[event_type])

        for cb in callbacks:
            try:
                cb(event_data)
            except Exception as e:
                logger.error(f"Error executing callback for event {event_type}: {e}")
