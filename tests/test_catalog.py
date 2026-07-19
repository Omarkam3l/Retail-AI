"""Tests for product catalog and loader."""
import pytest
import tempfile
import os
import numpy as np
from src.product_recognition.catalog import ProductCatalog
from src.product_recognition.catalog_loader import CatalogLoader
from src.product_recognition.types import ProductRecord


def test_catalog_operations():
    catalog = ProductCatalog()
    prod = ProductRecord(
        sku="SKU-001",
        name="Cola 330ml",
        brand="Coca-Cola",
        category="beverage",
        embedding=np.zeros(384, dtype=np.float32)
    )

    catalog.add_product(prod)
    assert catalog.get_by_sku("SKU-001") is not None
    assert catalog.version == 2

    # Search
    results = catalog.search_by_name("Cola")
    assert len(results) == 1

    results_cat = catalog.search_by_category("beverage")
    assert len(results_cat) == 1

    matrix = catalog.get_embeddings_matrix()
    assert matrix.shape == (1, 384)

    catalog.remove_product("SKU-001")
    assert catalog.get_by_sku("SKU-001") is None


def test_catalog_loader():
    catalog = ProductCatalog()
    prod = ProductRecord(
        sku="SKU-001",
        name="Cola 330ml",
        brand="Coca-Cola",
        category="beverage",
        embedding=np.zeros(384, dtype=np.float32)
    )
    catalog.add_product(prod)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "catalog.json")
        CatalogLoader.save_to_json(catalog, path)

        loaded = CatalogLoader.load_from_json(path)
        assert loaded.get_by_sku("SKU-001") is not None
        assert loaded.version == catalog.version
