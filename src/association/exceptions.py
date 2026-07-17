from src.common.exceptions import PipelineError

class AssociationError(PipelineError):
    """Raised when spatial, pose, or motion correlation tracking fails to resolve associations."""
    pass
