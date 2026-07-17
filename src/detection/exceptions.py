from src.common.exceptions import InferenceError

class ObjectDetectionError(InferenceError):
    """Raised when the object detection model fails to execute or load weights."""
    pass
