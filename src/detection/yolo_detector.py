import logging
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
import torch

from src.detection.interfaces import BaseDetector
from src.detection.exceptions import ObjectDetectionError
from src.common.types import DetectedObject, BoundingBox, ClassLabel

logger = logging.getLogger("YOLO11Detector")

class YOLO11Detector(BaseDetector):
    """Production-grade object detector using YOLO11s via Ultralytics."""

    # Map COCO string labels to domain ClassLabel enums
    COCO_LABEL_MAPPING = {
        "person": ClassLabel.PERSON,
        "backpack": ClassLabel.BACKPACK,
        "handbag": ClassLabel.HANDBAG,
        "suitcase": ClassLabel.SHOPPING_BASKET,
        "bottle": ClassLabel.SHELF_ITEM,
        "wine glass": ClassLabel.SHELF_ITEM,
        "cup": ClassLabel.SHELF_ITEM,
        "fork": ClassLabel.SHELF_ITEM,
        "knife": ClassLabel.SHELF_ITEM,
        "spoon": ClassLabel.SHELF_ITEM,
        "bowl": ClassLabel.SHELF_ITEM,
        "banana": ClassLabel.SHELF_ITEM,
        "apple": ClassLabel.SHELF_ITEM,
        "sandwich": ClassLabel.SHELF_ITEM,
        "orange": ClassLabel.SHELF_ITEM,
        "broccoli": ClassLabel.SHELF_ITEM,
        "carrot": ClassLabel.SHELF_ITEM,
        "hot dog": ClassLabel.SHELF_ITEM,
        "pizza": ClassLabel.SHELF_ITEM,
        "donut": ClassLabel.SHELF_ITEM,
        "cake": ClassLabel.SHELF_ITEM,
        "chair": ClassLabel.SHELF_ITEM,
        "book": ClassLabel.SHELF_ITEM,
        "cell phone": ClassLabel.SHELF_ITEM
    }

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
        fp16: bool = True
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._device_setting = device
        self._fp16 = fp16
        
        self._model: Optional[YOLO] = None
        self._device: str = "cpu"

    def initialize(self) -> None:
        """Loads weight parameters, configures execution device, and runs warm-up."""
        try:
            logger.info(f"Loading YOLO11 model weights from: {self._model_path}")
            self._model = YOLO(self._model_path)
            
            # Device selection
            if self._device_setting == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._device_setting
                
            logger.info(f"Executing YOLO11 inference on device: {self._device}")
            
            # Execute model warm-up
            self._warmup()
            logger.info("YOLO11 initialization and warm-up completed successfully.")
            
        except Exception as e:
            raise ObjectDetectionError(f"Failed to initialize YOLO11 detector: {e}") from e

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Runs the complete preprocessing, inference, and postprocessing pipeline.

        Args:
            frame: A NumPy array representing the decoded video frame.

        Returns:
            A list of detected domain object instances.
        """
        if self._model is None:
            raise ObjectDetectionError("Detector is not initialized. Call initialize() first.")

        try:
            # Preprocessing validation
            self._preprocess(frame)
            
            # Inference execution
            results = self._inference(frame)
            
            # Postprocessing & mapping
            return self._postprocess(results, frame.shape)
            
        except Exception as e:
            raise ObjectDetectionError(f"Error during frame detection execution: {e}") from e

    def benchmark(self, frames: List[np.ndarray], iterations: int = 10) -> Dict[str, Any]:
        """Measures detection throughput (FPS) and latency profile on a sample frame set."""
        if not frames:
            return {"fps": 0.0, "latency_ms": 0.0}
            
        logger.info(f"Starting object detection benchmark on {len(frames)} frames across {iterations} iterations...")
        latencies = []
        
        # Warmup loop
        for frame in frames[:5]:
            self.detect(frame)
            
        for _ in range(iterations):
            for frame in frames:
                start = time.perf_counter()
                self.detect(frame)
                latencies.append((time.perf_counter() - start) * 1000.0)
                
        avg_latency = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_latency
        
        return {
            "fps": fps,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        }

    def shutdown(self) -> None:
        """Deallocates model and GPU caches."""
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("YOLO11 resources released.")

    def _warmup(self) -> None:
        """Runs dummy frames through model to populate CUDA graphs and memory pools."""
        if self._model is None:
            return
            
        logger.info("Executing model warm-up iterations...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Perform 3 dummy runs
        for _ in range(3):
            self._model.predict(
                dummy_frame,
                device=self._device,
                half=self._fp16 and self._device == "cuda",
                verbose=False
            )

    def _preprocess(self, frame: np.ndarray) -> None:
        """Validates input frame dimensions and type integrity."""
        if not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Input frame must have 3 dimensions (BGR), got shape {frame.shape}")

    def _inference(self, frame: np.ndarray) -> List[Any]:
        """Runs the core model forward pass."""
        # Use predict API with FP16 options
        return self._model.predict(
            frame,
            device=self._device,
            conf=self._conf_threshold,
            half=self._fp16 and self._device == "cuda",
            verbose=False
        )

    def _postprocess(self, results: List[Any], frame_shape: Tuple[int, ...]) -> List[DetectedObject]:
        """Filters YOLO output boxes and maps coordinates to normalized values."""
        detected_objects: List[DetectedObject] = []
        h, w = frame_shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Class parsing
                class_id = int(box.cls[0])
                coco_label = self._model.names[class_id]
                
                # Check if this class is mapped in our domain labels
                class_label = self.COCO_LABEL_MAPPING.get(coco_label)
                if class_label is None:
                    continue
                    
                # Bounding box coordinates (denormalized xyxy)
                xyxy = box.xyxy[0].tolist()
                
                # Normalize coordinates relative to image resolution [0.0, 1.0]
                normalized_bbox = BoundingBox(
                    x_min=max(0.0, min(1.0, xyxy[0] / w)),
                    y_min=max(0.0, min(1.0, xyxy[1] / h)),
                    x_max=max(0.0, min(1.0, xyxy[2] / w)),
                    y_max=max(0.0, min(1.0, xyxy[3] / h))
                )
                
                confidence = float(box.conf[0])
                
                detected_objects.append(
                    DetectedObject(
                        class_label=class_label,
                        bbox=normalized_bbox,
                        confidence=confidence
                    )
                )
                
        return detected_objects
