from src.common.exceptions import InferenceError

class ObjectTrackingError(InferenceError):
    """Raised when the tracking algorithm fails to initialize or associate frames."""
    pass
