from src.common.exceptions import APIConnectionError

class AuthenticationError(APIConnectionError):
    """Raised when JWT token authentication or edge token validation fails."""
    pass

class RequestValidationError(APIConnectionError):
    """Raised when JSON payload models do not match schema specifications."""
    pass
