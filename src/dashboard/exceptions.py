from src.common.exceptions import PlatformException

class DashboardError(PlatformException):
    """Raised when dashboard template rendering or UI coordinate mapping fails."""
    pass
