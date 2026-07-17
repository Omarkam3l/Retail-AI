from abc import ABC, abstractmethod

class Lifecycle(ABC):
    """Abstract interface defining standard service lifecycle controls."""

    @abstractmethod
    def initialize(self) -> None:
        """Perform initial setup, allocate resources, or load model weights."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources, close connections, or deallocate buffers."""
        pass
