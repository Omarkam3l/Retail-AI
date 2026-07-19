"""Standard feature extraction cascade integrating preprocessor and model."""
import logging
from typing import List
import numpy as np
from src.product_recognition.interfaces import BaseFeatureExtractor, BaseEmbeddingModel
from src.product_recognition.preprocessing import ImagePreprocessor

logger = logging.getLogger("FeatureExtractor")


class FeatureExtractor(BaseFeatureExtractor):
    """Integrates Preprocessor and Embedding Model."""

    def __init__(self, model: BaseEmbeddingModel, image_size: int = 224) -> None:
        self._model = model
        self._preprocessor = ImagePreprocessor(target_size=image_size)

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        preprocessed = self._preprocessor.preprocess(crop)
        return self._model.get_embedding(preprocessed)

    def extract_features_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 384), dtype=np.float32)
        batch = self._preprocessor.preprocess_batch(crops)
        return self._model.get_embeddings_batch(batch)
