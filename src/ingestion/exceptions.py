from src.common.exceptions import PlatformException

class VideoIngestionError(PlatformException):
    """Base exception for all video ingestion and processing errors."""
    pass

class VideoSourceError(VideoIngestionError):
    """Raised when a video source fails to open, read, or release resources."""
    pass

class BufferError(VideoIngestionError):
    """Base exception for circular buffer operations."""
    pass

class BufferOverflowError(BufferError):
    """Raised when attempting to push to a full circular buffer under strict bounds."""
    pass

class BufferUnderflowError(BufferError):
    """Raised when attempting to retrieve frames from an empty buffer."""
    pass
