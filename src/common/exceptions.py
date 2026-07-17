class PlatformException(Exception):
    """Base exception class for all errors in the Retail AI Surveillance Platform."""
    pass

class ConfigurationError(PlatformException):
    """Raised when configuration parameters are invalid or missing."""
    pass

class PipelineError(PlatformException):
    """Raised when there is an error in the execution of the inference pipeline."""
    pass

class InferenceError(PlatformException):
    """Raised when object detection, tracking, or pose models encounter runtime failures."""
    pass

class DataStorageError(PlatformException):
    """Raised when database or file operations fail."""
    pass

class APIConnectionError(PlatformException):
    """Raised when connection to cloud APIs or WebSocket endpoints fails."""
    pass
