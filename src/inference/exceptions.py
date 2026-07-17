from src.common.exceptions import PipelineError

class InferencePipelineError(PipelineError):
    """Raised when frame synchronization, thread communication, or stage cascades fail."""
    pass
