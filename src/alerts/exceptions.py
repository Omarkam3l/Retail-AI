from src.common.exceptions import PipelineError

class AlertEngineError(PipelineError):
    """Raised when face blurring, video encoding, or alert validation fails."""
    pass
