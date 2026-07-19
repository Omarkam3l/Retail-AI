"""Product catalog manager keeping product records and embeddings indices."""
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from src.product_recognition.interfaces import BaseProductCatalog
from src.product_recognition.types import ProductRecord
from src.product_recognition.exceptions import CatalogError

logger = logging.getLogger("ProductCatalog")


class ProductCatalog(BaseProductCatalog):
    """In-memory product catalog with indexing support."""

    def __init__(self) -> None:
        self._products: Dict[str, ProductRecord] = {}
        self._version: int = 1

    def add_product(self, product: ProductRecord) -> None:
        if not product.sku:
            raise CatalogError("Product SKU cannot be empty.")
        self._products[product.sku] = product
        self._version += 1
        logger.info(f"Product added to catalog: {product.name} (SKU: {product.sku})")

    def remove_product(self, sku: str) -> None:
        if sku in self._products:
            del self._products[sku]
            self._version += 1
            logger.info(f"Product removed from catalog: SKU {sku}")
        else:
            raise CatalogError(f"SKU {sku} not found in catalog.")

    def get_by_sku(self, sku: str) -> Optional[ProductRecord]:
        return self._products.get(sku)

    def search_by_name(self, query: str) -> List[ProductRecord]:
        query_lower = query.lower()
        return [
            p for p in self._products.values()
            if query_lower in p.name.lower() or query_lower in p.brand.lower()
        ]

    def search_by_category(self, category: str) -> List[ProductRecord]:
        cat_lower = category.lower()
        return [p for p in self._products.values() if cat_lower == p.category.lower()]

    def list_all(self) -> List[ProductRecord]:
        return list(self._products.values())

    def get_embeddings_matrix(self) -> np.ndarray:
        if not self._products:
            return np.empty((0, 384), dtype=np.float32)  # DINOv2 vit_s features output
        return np.vstack([p.embedding for p in self._products.values()])

    @property
    def version(self) -> int:
        return self._version
