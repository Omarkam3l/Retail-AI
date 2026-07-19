"""Cosine similarity index matcher support Top-K search and threshold filtering."""
import logging
from typing import List
import numpy as np
from src.product_recognition.interfaces import BaseSimilarityEngine, BaseProductCatalog
from src.product_recognition.types import MatchResult

logger = logging.getLogger("SimilarityEngine")


class SimilarityEngine(BaseSimilarityEngine):
    """Computes cosine similarities between inputs and reference vectors."""

    def __init__(self, catalog: BaseProductCatalog, threshold: float = 0.7) -> None:
        self._catalog = catalog
        self._threshold = threshold

    def find_similar(self, embedding: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """Calculates similarities using matrix dot product for normalized vectors."""
        products = self._catalog.list_all()
        if not products:
            return []

        matrix = self._catalog.get_embeddings_matrix()
        if matrix.shape[0] == 0:
            return []

        # normalized vectors dot product = cosine similarity
        similarities = np.dot(matrix, embedding)

        # argsort in descending order
        indices = np.argsort(similarities)[::-1]

        results = []
        for rank, idx in enumerate(indices[:top_k], 1):
            sim = float(similarities[idx])
            prod = products[idx]

            # Even if below threshold, return in lists (matcher/unknown handler filters)
            results.append(
                MatchResult(
                    sku=prod.sku,
                    name=prod.name,
                    brand=prod.brand,
                    category=prod.category,
                    similarity=sim,
                    confidence=sim,  # Initial raw confidence matches similarity
                    rank=rank
                )
            )
        return results
