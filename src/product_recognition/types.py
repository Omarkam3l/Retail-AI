"""Data structures and types for the Product Recognition module."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class ProductRecord:
    sku: str
    name: str
    brand: str
    category: str
    embedding: np.ndarray
    reference_images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "reference_images": self.reference_images,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductRecord":
        return cls(
            sku=data["sku"],
            name=data["name"],
            brand=data["brand"],
            category=data["category"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            reference_images=data.get("reference_images", []),
            metadata=data.get("metadata", {})
        )


@dataclass(frozen=True)
class MatchResult:
    sku: str
    name: str
    brand: str
    category: str
    similarity: float
    confidence: float
    rank: int


@dataclass
class RecognitionResult:
    track_id: int
    recognized: bool
    sku: Optional[str] = None
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    similarity: float = 0.0
    confidence: float = 0.0
    matches: List[MatchResult] = field(default_factory=list)
