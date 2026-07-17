import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.detection.yolo_detector import YOLO11Detector
from src.detection.exceptions import ObjectDetectionError
from src.common.types import ClassLabel, BoundingBox

def test_yolo_detector_invalid_init():
    detector = YOLO11Detector(model_path="nonexistent.pt")
    with pytest.raises(ObjectDetectionError):
        # Should raise error because file doesn't exist
        detector.initialize()


@patch("src.detection.yolo_detector.YOLO")
def test_yolo_detector_mocked_inference(mock_yolo_class):
    # Setup mock YOLO instance
    mock_model_instance = MagicMock()
    # Mock class labels mapping
    mock_model_instance.names = {0: "person", 24: "backpack", 26: "handbag", 39: "bottle"}
    mock_yolo_class.return_value = mock_model_instance

    # Mock predictions output
    mock_box = MagicMock()
    mock_box.cls = [0]       # class 0 ("person")
    mock_box.conf = [0.85]   # confidence
    mock_box.xyxy = [np.array([100, 150, 200, 300])]  # bbox coordinates

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]

    # Initialize detector
    detector = YOLO11Detector(model_path="mock_yolo.pt", device="cpu")
    detector.initialize()

    # Create dummy frame (size: 640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = detector.detect(frame)
    
    assert len(detections) == 1
    det = detections[0]
    
    # Assert correct domain class mapping
    assert det.class_label == ClassLabel.PERSON
    assert det.confidence == 0.85
    
    # Assert coordinates are normalized [0.0, 1.0] relative to frame dimensions (640x480)
    assert pytest.approx(det.bbox.x_min) == 100 / 640
    assert pytest.approx(det.bbox.y_min) == 150 / 480
    assert pytest.approx(det.bbox.x_max) == 200 / 640
    assert pytest.approx(det.bbox.y_max) == 300 / 480


@patch("src.detection.yolo_detector.YOLO")
def test_yolo_detector_unmapped_class(mock_yolo_class):
    mock_model_instance = MagicMock()
    mock_model_instance.names = {15: "cat"} # Class 15 "cat" is not in our surveillance domain
    mock_yolo_class.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.cls = [15]
    mock_box.conf = [0.9]
    mock_box.xyxy = [np.array([10, 10, 50, 50])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]

    detector = YOLO11Detector(model_path="mock_yolo.pt", device="cpu")
    detector.initialize()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    # "cat" detections should be filtered/skipped entirely
    assert len(detections) == 0
