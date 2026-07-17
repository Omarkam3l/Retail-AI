import logging
import threading
from typing import Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("ModelLifecycleManager")

class ModelLifecycleManager:
    """Manages downloading, caching, warming up, and hot-swapping AI models thread-safely."""

    def __init__(self) -> None:
        # Key: model_id/name -> Loaded Model Instance
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_model(self, model_name: str) -> Optional[Any]:
        """Retrieves a loaded model instance from cache."""
        with self._lock:
            return self._models.get(model_name)

    def load_model(self, model_name: str, model_path: str) -> Any:
        """Loads and caches a model from the specified path."""
        with self._lock:
            logger.info(f"Loading model '{model_name}' from path: {model_path}")
            model = YOLO(model_path)
            
            # Run a dummy dry-run to warm up weights
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                model.predict(dummy_img, verbose=False)
                logger.info(f"Model '{model_name}' successfully warmed up.")
            except Exception as e:
                logger.warning(f"Could not dry-run warm up model '{model_name}': {e}")
                
            self._models[model_name] = model
            return model

    def hot_reload_model(self, model_name: str, new_path: str) -> None:
        """Hot-reloads a model instance at runtime under thread-safe locking."""
        logger.info(f"Hot-reloading model '{model_name}' with new path: {new_path}")
        new_model = YOLO(new_path)
        
        # Warm up new instance before swapping references
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            new_model.predict(dummy_img, verbose=False)
        except Exception as e:
            logger.warning(f"Dry-run warm up for hot-reloaded model '{model_name}' failed: {e}")

        # Swap references under lock
        with self._lock:
            self._models[model_name] = new_model
            logger.info(f"Model '{model_name}' reference successfully hot-swapped.")

    def clear(self) -> None:
        with self._lock:
            self._models.clear()
