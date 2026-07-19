"""Evaluation and benchmarking runner measuring Top-K accuracies."""
import time
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from src.product_recognition.recognition_engine import ProductRecognitionEngine

logger = logging.getLogger("RecognitionBenchmark")


class RecognitionBenchmark:
    """Benchmarks accuracy, latency, and throughput on test data pools."""

    def __init__(self, engine: ProductRecognitionEngine) -> None:
        self._engine = engine

    def run_benchmark(
        self,
        test_images: List[np.ndarray],
        ground_truth_skus: List[str]
    ) -> Dict[str, Any]:
        """Evaluates model performance against target catalog."""
        from src.common.types import BoundingBox
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        
        total = len(test_images)
        top1_hits = 0
        top5_hits = 0
        latencies = []
        unknown_hits = 0

        logger.info(f"Starting product recognition benchmark on {total} samples...")

        for idx, (img, gt_sku) in enumerate(zip(test_images, ground_truth_skus)):
            start = time.perf_counter()
            res = self._engine.process_object(img, bbox, track_id=idx, detection_confidence=1.0)
            latencies.append((time.perf_counter() - start) * 1000)

            # Evaluate top-k match list
            match_skus = [m.sku for m in res.matches]
            
            if res.recognized and res.sku == gt_sku:
                top1_hits += 1

            if gt_sku in match_skus:
                top5_hits += 1

            if not res.recognized and gt_sku == "UNKNOWN":
                unknown_hits += 1

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            "total_samples": total,
            "top1_accuracy": top1_hits / total if total > 0 else 0.0,
            "top5_accuracy": top5_hits / total if total > 0 else 0.0,
            "average_latency_ms": avg_latency,
            "throughput_fps": fps,
            "unknown_rate": unknown_hits / total if total > 0 else 0.0
        }
