"""DINOv2 embedding model wrapper with PyTorch runtime."""
import logging
from typing import List, Optional
import numpy as np
from src.product_recognition.interfaces import BaseEmbeddingModel
from src.product_recognition.exceptions import EmbeddingError

logger = logging.getLogger("EmbeddingModel")


class DINOv2EmbeddingModel(BaseEmbeddingModel):
    """Wraps DINOv2 ViT model logic for embedding generation."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "auto") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._torch = None

    def initialize(self) -> None:
        """Loads torch model hubs safely."""
        try:
            import torch
            self._torch = torch
            
            # Select device
            if self._device == "auto":
                self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._actual_device = self._device

            logger.info(f"Loading model '{self._model_name}' on device '{self._actual_device}'...")
            
            # Force model hub load from local/cache if possible
            self._model = torch.hub.load("facebookresearch/dinov2", self._model_name, trust_repo=True)
            self._model.to(self._actual_device)
            self._model.eval()
            
            logger.info(f"Successfully loaded model '{self._model_name}'.")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize DINOv2 model: {e}")

    def get_embedding(self, chw_image: np.ndarray) -> np.ndarray:
        """Generates embedding for a preprocessed image."""
        if self._model is None:
            raise EmbeddingError("Model has not been initialized.")

        try:
            torch = self._torch
            tensor = torch.from_numpy(chw_image).unsqueeze(0).to(self._actual_device)

            with torch.no_grad():
                features = self._model(tensor)
                
            # L2 normalize
            norm_features = torch.nn.functional.normalize(features, dim=1)
            return norm_features.cpu().numpy()[0]
        except Exception as e:
            raise EmbeddingError(f"Embedding extraction failed: {e}")

    def get_embeddings_batch(self, batch_chw_images: np.ndarray) -> np.ndarray:
        """Generates embedding matrix for a batch of preprocessed images."""
        if self._model is None:
            raise EmbeddingError("Model has not been initialized.")

        if batch_chw_images.shape[0] == 0:
            return np.empty((0, 384), dtype=np.float32)

        try:
            torch = self._torch
            tensor = torch.from_numpy(batch_chw_images).to(self._actual_device)

            with torch.no_grad():
                features = self._model(tensor)

            norm_features = torch.nn.functional.normalize(features, dim=1)
            return norm_features.cpu().numpy()
        except Exception as e:
            raise EmbeddingError(f"Batch embedding extraction failed: {e}")
