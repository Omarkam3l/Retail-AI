"""Tests for preprocessor and feature extractor using mock embedding models."""
import pytest
import numpy as np
from typing import List
from src.product_recognition.preprocessing import ImagePreprocessor
from src.product_recognition.feature_extractor import FeatureExtractor
from src.product_recognition.interfaces import BaseEmbeddingModel


class MockEmbeddingModel(BaseEmbeddingModel):
    def initialize(self) -> None:
        pass
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        return np.ones(384, dtype=np.float32) * 0.5
    def get_embeddings_batch(self, images: List[np.ndarray]) -> np.ndarray:
        n = images.shape[0] if hasattr(images, "shape") else len(images)
        return np.ones((n, 384), dtype=np.float32) * 0.5


def test_preprocessor():
    preprocessor = ImagePreprocessor(target_size=224)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    chw = preprocessor.preprocess(img)
    assert chw.shape == (3, 224, 224)


def test_feature_extractor():
    mock_model = MockEmbeddingModel()
    extractor = FeatureExtractor(mock_model)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = extractor.extract_features(img)
    assert emb.shape == (384,)
