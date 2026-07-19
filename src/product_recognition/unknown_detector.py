"""Saves crops of unknown products for later tagging/fine-tuning."""
import os
import uuid
import cv2
import logging
import json
import numpy as np
from src.product_recognition.types import RecognitionResult

logger = logging.getLogger("UnknownDetector")


class UnknownDetector:
    """Detects and logs unknown product events."""

    def __init__(self, output_dir: str = "data/unknowns") -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_unknown(self, crop: np.ndarray, result: RecognitionResult, embedding: np.ndarray) -> str:
        """Saves crop image and embedding vector file under unique ID."""
        unknown_id = str(uuid.uuid4())
        
        # Save image
        img_path = os.path.join(self._output_dir, f"{unknown_id}.jpg")
        cv2.imwrite(img_path, crop)

        # Save embedding and similarity metadata
        meta_path = os.path.join(self._output_dir, f"{unknown_id}.json")
        meta = {
            "id": unknown_id,
            "best_similarity": float(result.similarity),
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Unknown product crop saved to {img_path} (similarity: {result.similarity:.2f})")
        return unknown_id
