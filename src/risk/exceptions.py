from src.common.exceptions import PipelineError

class RiskEngineError(PipelineError):
    """Raised when risk aggregation or score decay operations fail."""
    pass
