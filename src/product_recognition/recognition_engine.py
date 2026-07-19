"""Main facade integrating crop processing, matching, and caching."""
import time
import logging
from typing import List, Tuple
import numpy as np

from src.product_recognition.interfaces import BaseFeatureExtractor, BaseProductMatcher
from src.product_recognition.crop_processor import CropProcessor
from src.product_recognition.types import RecognitionResult, ProductRecord
from src.product_recognition.cache import EmbeddingCache
from src.product_recognition.unknown_detector import UnknownDetector
from src.product_recognition.confidence import ConfidenceEngine
from src.product_recognition.metrics import PerformanceTracker
from src.common.types import FrameMetadata, BoundingBox

logger = logging.getLogger("ProductRecognitionEngine")


class ProductRecognitionEngine:
    """Coordinates crop extraction, embedding generation, caching, matching, and fusing."""

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        matcher: BaseProductMatcher,
        cache_size: int = 1000,
        unknowns_dir: str = "data/unknown_products",
        similarity_threshold: float = 0.7
    ) -> None:
        self._extractor = extractor
        self._matcher = matcher
        self._cache = EmbeddingCache(maxsize=cache_size)
        self._unknown_detector = UnknownDetector(output_dir=unknowns_dir)
        self._confidence_engine = ConfidenceEngine()
        self._tracker = PerformanceTracker()
        self._threshold = similarity_threshold

    def initialize(self) -> None:
        """Initializes the underlying model weights and hardware resources."""
        if hasattr(self._extractor, "_model") and hasattr(self._extractor._model, "initialize"):
            self._extractor._model.initialize()
        elif hasattr(self._extractor, "initialize"):
            self._extractor.initialize()

    def process_object(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        track_id: int,
        detection_confidence: float
    ) -> RecognitionResult:
        """Processes a single tracked object bounding box."""
        start = time.perf_counter()
        
        # Check cache
        cache_key = f"track_{track_id}"
        embedding = self._cache.get(cache_key)

        crop = CropProcessor.extract_crop(frame, bbox)
        if crop.size == 0:
            return RecognitionResult(track_id=track_id, recognized=False)

        if embedding is None:
            # Generate embedding
            try:
                embedding = self._extractor.extract_features(crop)
                self._cache.put(cache_key, embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return RecognitionResult(track_id=track_id, recognized=False)

        # Find match
        res = self._matcher.match(embedding)
        res.track_id = track_id

        # Calculate final fused confidence score
        if res.matches:
            best_match = res.matches[0]
            fused_conf = self._confidence_engine.fuse_confidence(
                best_match.similarity,
                detection_confidence,
                crop
            )
            res.confidence = fused_conf

        # Process unknown items
        if not res.recognized:
            self._unknown_detector.process_unknown(crop, res, embedding)
            self._tracker.record_inference(
                (time.perf_counter() - start) * 1000,
                is_unknown=True
            )
        else:
            self._tracker.record_inference(
                (time.perf_counter() - start) * 1000,
                is_unknown=False
            )

        return res
