"""Product matcher using Top-K search and matching logic."""
import logging
from typing import List, Optional
import numpy as np
from src.product_recognition.interfaces import BaseProductMatcher, BaseSimilarityEngine
from src.product_recognition.types import RecognitionResult, MatchResult

logger = logging.getLogger("ProductMatcher")


class ProductMatcher(BaseProductMatcher):
    """Retrieves best candidate product matches from similarity engines."""

    def __init__(self, similarity_engine: BaseSimilarityEngine, threshold: float = 0.7) -> None:
        self._similarity_engine = similarity_engine
        self._threshold = threshold

    def match(self, embedding: np.ndarray) -> RecognitionResult:
        """Finds similarity matches and creates standard RecognitionResult."""
        matches = self._similarity_engine.find_similar(embedding, top_k=5)

        if not matches:
            return RecognitionResult(track_id=0, recognized=False)

        best_match = matches[0]

        if best_match.similarity >= self._threshold:
            return RecognitionResult(
                track_id=0,
                recognized=True,
                sku=best_match.sku,
                name=best_match.name,
                brand=best_match.brand,
                category=best_match.category,
                similarity=best_match.similarity,
                confidence=best_match.confidence,
                matches=matches
            )

        # Found candidates but below threshold (unknown product)
        return RecognitionResult(
            track_id=0,
            recognized=False,
            similarity=best_match.similarity,
            confidence=best_match.confidence,
            matches=matches
        )
