from src.common.exceptions import PipelineError

class BehaviorEngineError(PipelineError):
    """Raised when spatial-temporal sequence checks or state transitions fail."""
    pass
