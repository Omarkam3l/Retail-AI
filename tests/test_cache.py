"""Tests for embedding LRU cache."""
import pytest
import numpy as np
from src.product_recognition.cache import EmbeddingCache


def test_embedding_cache():
    cache = EmbeddingCache(maxsize=2)
    v1 = np.ones(384, dtype=np.float32)
    v2 = np.ones(384, dtype=np.float32) * 2
    v3 = np.ones(384, dtype=np.float32) * 3

    cache.put("k1", v1)
    cache.put("k2", v2)

    assert np.array_equal(cache.get("k1"), v1)

    # Put v3, evict k2 (k1 was accessed, so k2 is oldest)
    cache.put("k3", v3)
    assert cache.get("k2") is None
    assert cache.get("k3") is not None

    h, m = cache.stats
    assert h > 0
