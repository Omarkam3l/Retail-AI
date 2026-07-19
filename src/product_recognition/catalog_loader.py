"""Loads and saves catalog index files to disk in JSON format."""
import os
import json
import logging
from typing import Dict, Any
from src.product_recognition.catalog import ProductCatalog
from src.product_recognition.types import ProductRecord
from src.product_recognition.exceptions import CatalogError

logger = logging.getLogger("CatalogLoader")


class CatalogLoader:
    """Manages serialization and deserialization of the ProductCatalog."""

    @staticmethod
    def load_from_json(filepath: str) -> ProductCatalog:
        """Loads catalog from a JSON file."""
        catalog = ProductCatalog()
        if not os.path.exists(filepath):
            logger.warning(f"Catalog file not found: {filepath}. Returning empty catalog.")
            return catalog

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            products_data = data.get("products", [])
            for p_dict in products_data:
                record = ProductRecord.from_dict(p_dict)
                catalog.add_product(record)

            catalog._version = data.get("version", 1)
            logger.info(f"Loaded {len(products_data)} products from {filepath}")
            return catalog
        except Exception as e:
            raise CatalogError(f"Failed to load catalog from JSON: {e}")

    @staticmethod
    def save_to_json(catalog: ProductCatalog, filepath: str) -> None:
        """Saves catalog to a JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            data = {
                "version": catalog.version,
                "products": [p.to_dict() for p in catalog.list_all()]
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved catalog (version {catalog.version}) to {filepath}")
        except Exception as e:
            raise CatalogError(f"Failed to save catalog to JSON: {e}")
