"""Inference batch aggregator maximizing throughput."""
import logging
from typing import List, Tuple
import numpy as np
from src.product_recognition.interfaces import BaseFeatureExtractor

logger = logging.getLogger("BatchProcessor")


class BatchProcessor:
    """Aggregates parallel crops requests and feeds them as a single batch."""

    def __init__(self, extractor: BaseFeatureExtractor, batch_size: int = 8) -> None:
        self._extractor = extractor
        self._batch_size = batch_size

    def process_crops(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Splits list of crop arrays into optimized chunks for feature extraction."""
        if not crops:
            return []

        embeddings = []
        for i in range(0, len(crops), self._batch_size):
            chunk = crops[i:i + self._batch_size]
            chunk_embeddings = self._extractor.extract_features_batch(chunk)
            for j in range(chunk_embeddings.shape[0]):
                embeddings.append(chunk_embeddings[j])

        return embeddings
