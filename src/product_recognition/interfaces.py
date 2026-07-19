"""Abstract interfaces defining the contracts for Product Recognition components."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from src.product_recognition.types import ProductRecord, MatchResult, RecognitionResult


class BaseEmbeddingModel(ABC):
    """Interface for target neural network inference wrapper."""

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_embeddings_batch(self, images: List[np.ndarray]) -> np.ndarray:
        pass


class BaseFeatureExtractor(ABC):
    """Interface for image preprocessing and normalization cascade."""

    @abstractmethod
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def extract_features_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        pass


class BaseProductCatalog(ABC):
    """Interface for product records index storage."""

    @abstractmethod
    def add_product(self, product: ProductRecord) -> None:
        pass

    @abstractmethod
    def remove_product(self, sku: str) -> None:
        pass

    @abstractmethod
    def get_by_sku(self, sku: str) -> Optional[ProductRecord]:
        pass

    @abstractmethod
    def search_by_name(self, query: str) -> List[ProductRecord]:
        pass

    @abstractmethod
    def search_by_category(self, category: str) -> List[ProductRecord]:
        pass

    @abstractmethod
    def list_all(self) -> List[ProductRecord]:
        pass

    @abstractmethod
    def get_embeddings_matrix(self) -> np.ndarray:
        pass


class BaseSimilarityEngine(ABC):
    """Interface for vector indexing and cosine similarity calculation."""

    @abstractmethod
    def find_similar(self, embedding: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        pass


class BaseProductMatcher(ABC):
    """Interface for ranking and selecting matches."""

    @abstractmethod
    def match(self, embedding: np.ndarray) -> RecognitionResult:
        pass
