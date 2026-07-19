"""Tests for matching engine."""
import pytest
import numpy as np
from src.product_recognition.catalog import ProductCatalog
from src.product_recognition.similarity import SimilarityEngine
from src.product_recognition.matcher import ProductMatcher
from src.product_recognition.types import ProductRecord


def test_product_matcher():
    catalog = ProductCatalog()
    p1 = ProductRecord("sku1", "item1", "brand1", "cat1", np.array([1.0, 0.0], dtype=np.float32))
    catalog.add_product(p1)

    sim = SimilarityEngine(catalog)
    matcher = ProductMatcher(sim, threshold=0.8)

    # exact match
    res = matcher.match(np.array([1.0, 0.0], dtype=np.float32))
    assert res.recognized is True
    assert res.sku == "sku1"

    # low similarity match
    res_low = matcher.match(np.array([0.1, 0.9], dtype=np.float32))
    assert res_low.recognized is False
