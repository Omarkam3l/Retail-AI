"""Tests for similarity matching engine."""
import pytest
import numpy as np
from src.product_recognition.catalog import ProductCatalog
from src.product_recognition.similarity import SimilarityEngine
from src.product_recognition.types import ProductRecord


def test_similarity_search():
    catalog = ProductCatalog()
    p1 = ProductRecord("sku1", "item1", "brand1", "cat1", np.array([1.0, 0.0], dtype=np.float32))
    p2 = ProductRecord("sku2", "item2", "brand2", "cat2", np.array([0.0, 1.0], dtype=np.float32))
    catalog.add_product(p1)
    catalog.add_product(p2)

    engine = SimilarityEngine(catalog)
    # Search with p1 embedding
    matches = engine.find_similar(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert len(matches) == 2
    assert matches[0].sku == "sku1"
    assert matches[0].similarity == pytest.approx(1.0)
