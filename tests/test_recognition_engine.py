"""Tests for overall recognition coordinator engine."""
import pytest
import numpy as np
from src.product_recognition.catalog import ProductCatalog
from src.product_recognition.similarity import SimilarityEngine
from src.product_recognition.matcher import ProductMatcher
from src.product_recognition.feature_extractor import FeatureExtractor
from src.product_recognition.recognition_engine import ProductRecognitionEngine
from src.product_recognition.types import ProductRecord
from src.common.types import BoundingBox
from src.product_recognition.interfaces import BaseEmbeddingModel
from typing import List

class MockEmbeddingModel(BaseEmbeddingModel):
    def initialize(self) -> None:
        pass
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        return np.ones(384, dtype=np.float32) * 0.5
    def get_embeddings_batch(self, images: List[np.ndarray]) -> np.ndarray:
        n = images.shape[0] if hasattr(images, "shape") else len(images)
        return np.ones((n, 384), dtype=np.float32) * 0.5


def test_recognition_engine():
    catalog = ProductCatalog()
    p1 = ProductRecord("sku1", "item1", "brand1", "cat1", np.ones(384, dtype=np.float32) * 0.5)
    catalog.add_product(p1)

    mock_model = MockEmbeddingModel()
    extractor = FeatureExtractor(mock_model)
    sim = SimilarityEngine(catalog)
    matcher = ProductMatcher(sim, threshold=0.8)

    engine = ProductRecognitionEngine(extractor, matcher)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)

    res = engine.process_object(frame, bbox, track_id=1, detection_confidence=0.9)
    assert res.track_id == 1
