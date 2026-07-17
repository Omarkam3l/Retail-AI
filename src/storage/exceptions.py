from src.common.exceptions import DataStorageError

class DatabaseConnectionError(DataStorageError):
    """Raised when the database connection pool fails to initialize or times out."""
    pass

class QueryExecutionError(DataStorageError):
    """Raised when SQL queries or transaction commits fail."""
    pass
