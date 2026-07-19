"""Configuration validation definitions using Pydantic."""
from pydantic import BaseModel, Field
from typing import Dict, Any, List


class ProductRecognitionConfig(BaseModel):
    embedding_model: str = Field(default="dinov2_vits14")
    device: str = Field(default="auto")
    batch_size: int = Field(default=8, ge=1)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1)
    cache_size: int = Field(default=1000, ge=0)
    image_size: int = Field(default=224, ge=32)
    catalog_path: str = Field(default="data/product_catalog.json")
    unknowns_dir: str = Field(default="data/unknown_products")
    quality_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
