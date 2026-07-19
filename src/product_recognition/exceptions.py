"""Module-specific exceptions for product recognition."""


class ProductRecognitionError(Exception):
    """Base exception for all product recognition failures."""
    pass


class CatalogError(ProductRecognitionError):
    """Raised when catalog loading, saving, or updating fails."""
    pass


class EmbeddingError(ProductRecognitionError):
    """Raised when feature extraction or model inference fails."""
    pass


class SimilarityError(ProductRecognitionError):
    """Raised when vector indexing or similarity matching fails."""
    pass
