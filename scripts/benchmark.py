import time
import os
import psutil
import numpy as np
from unittest.mock import MagicMock, patch

from src.inference.orchestrator import PipelineOrchestrator
from src.detection.yolo_detector import YOLO11Detector
from src.tracking.adapter import ByteTrackAdapter
from src.association.engine import ObjectAssociationEngine
from src.behavior.engine import BehaviorEngine
from src.behavior.rules import PocketConcealmentRule

# Try to import torch for CUDA VRAM checking
try:
    import torch
except ImportError:
    torch = None

@patch("src.detection.yolo_detector.YOLO")
def run_benchmark(mock_yolo_class):
    print("==================================================")
    print("Starting Retail AI Platform Performance Benchmark")
    print("==================================================")

    # 1. Setup mocked YOLO weights
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "person", 39: "bottle"}
    mock_yolo_class.return_value = mock_model_instance

    box_person = MagicMock()
    box_person.cls = [0]
    box_person.conf = [0.9]
    box_person.xyxy = [np.array([100, 100, 200, 350])]

    mock_result = MagicMock()
    mock_result.boxes = [box_person]
    mock_model_instance.predict.return_value = [mock_result]

    # Initialize components
    detector = YOLO11Detector(model_path="mock.pt", device="cpu")
    tracker = ByteTrackAdapter(track_threshold=0.25)
    association_engine = ObjectAssociationEngine(proximity_threshold=0.5, persistence_threshold=2)
    behavior_engine = BehaviorEngine()
    behavior_engine.register_rule(PocketConcealmentRule())

    orchestrator = PipelineOrchestrator(
        camera_id="cam_benchmark",
        detector=detector,
        tracker=tracker,
        association_engine=association_engine,
        behavior_engine=behavior_engine
    )
    orchestrator.initialize()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 100

    print(f"Running pipeline inference on {num_frames} frames...")
    start_time = time.perf_counter()

    for i in range(num_frames):
        orchestrator.process_frame(frame, frame_index=i, timestamp_ms=float(i * 1000.0))

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    fps = num_frames / elapsed
    avg_latency = (elapsed / num_frames) * 1000.0

    # Retrieve memory footprint
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024.0 * 1024.0)

    print("\n---------------- RESULTS ----------------")
    print(f"FPS Processing Throughput: {fps:.2f} frames/sec")
    print(f"Average Frame Latency:     {avg_latency:.2f} ms/frame")
    print(f"System RAM Memory Usage:   {ram_usage_mb:.2f} MB")

    if torch is not None and torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        print(f"CUDA VRAM Allocation:      {vram_usage:.2f} MB")
    else:
        print("CUDA VRAM Allocation:      N/A (CPU Only)")

    # Reset
    orchestrator.shutdown()
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
