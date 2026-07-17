import logging
import time
from typing import List, Dict, Any, Callable
from src.evaluation.types import (
    GroundTruthAnnotation, PredictionRecord, OverallEvaluationResult
)
from src.evaluation.metrics.overall import OverallMetricsCalculator

logger = logging.getLogger("BenchmarkRunner")

class BenchmarkRunner:
    """Runs evaluations across multiple datasets, model versions, and threshold configurations."""

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def run_benchmark(
        self,
        name: str,
        ground_truths: List[GroundTruthAnnotation],
        predictions: List[PredictionRecord],
        model_version: str = "default",
        config_label: str = "default"
    ) -> OverallEvaluationResult:
        """Runs a single evaluation benchmark and stores the result."""
        start = time.perf_counter()
        calc = OverallMetricsCalculator()
        result = calc.compute(ground_truths, predictions)
        elapsed = time.perf_counter() - start

        entry = {
            "name": name,
            "model_version": model_version,
            "config_label": config_label,
            "execution_time_seconds": elapsed,
            "detection_f1": result.detection.f1 if result.detection else 0.0,
            "tracking_mota": result.tracking.mota if result.tracking else 0.0,
            "result": result
        }
        self._results.append(entry)
        logger.info(f"Benchmark '{name}' completed in {elapsed:.2f}s.")
        return result

    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """Returns a comparison table of all benchmark results."""
        table = []
        for r in self._results:
            table.append({
                "name": r["name"],
                "model": r["model_version"],
                "config": r["config_label"],
                "det_f1": f"{r['detection_f1']:.4f}",
                "trk_mota": f"{r['tracking_mota']:.4f}",
                "time_s": f"{r['execution_time_seconds']:.2f}"
            })
        return table

    def clear(self) -> None:
        self._results.clear()
