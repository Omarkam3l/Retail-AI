import logging
from typing import List, Dict, Any

logger = logging.getLogger("EvaluationVisualization")

class EvaluationVisualizer:
    """Generates visual evaluation artifacts. Uses matplotlib when available, falls back to text summaries."""

    def plot_precision_recall_curve(self, precisions: List[float], recalls: List[float],
                                    output_path: str = "pr_curve.png") -> str:
        """Generates a precision-recall curve plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recalls, precisions, "b-", linewidth=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"PR curve saved to {output_path}.")
            return output_path
        except ImportError:
            logger.warning("matplotlib not available. Skipping PR curve plot.")
            return ""

    def plot_confusion_matrix(self, matrix: Dict[str, Dict[str, int]], labels: List[str],
                               output_path: str = "confusion_matrix.png") -> str:
        """Generates a confusion matrix heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            n = len(labels)
            data = np.zeros((n, n))
            for i, actual in enumerate(labels):
                for j, predicted in enumerate(labels):
                    data[i][j] = matrix.get(actual, {}).get(predicted, 0)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap="Blues")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            fig.colorbar(im)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix saved to {output_path}.")
            return output_path
        except ImportError:
            logger.warning("matplotlib not available. Skipping confusion matrix plot.")
            return ""

    def plot_threshold_comparison(self, results: List[Dict[str, Any]],
                                   output_path: str = "threshold_comparison.png") -> str:
        """Generates a bar chart comparing threshold configurations."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = [f"{r.get('detection_confidence', '?')}" for r in results]
            scores = [r.get("score", 0) for r in results]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(labels, scores, color="steelblue")
            ax.set_xlabel("Detection Confidence Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Threshold Comparison")
            ax.grid(True, alpha=0.3, axis="y")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Threshold comparison saved to {output_path}.")
            return output_path
        except ImportError:
            logger.warning("matplotlib not available. Skipping threshold comparison plot.")
            return ""

    def plot_latency_chart(self, stage_latencies: Dict[str, float],
                            output_path: str = "latency_chart.png") -> str:
        """Generates a horizontal bar chart of pipeline stage latencies."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            stages = list(stage_latencies.keys())
            values = list(stage_latencies.values())

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(stages, values, color="coral")
            ax.set_xlabel("Latency (ms)")
            ax.set_title("Pipeline Stage Latencies")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Latency chart saved to {output_path}.")
            return output_path
        except ImportError:
            logger.warning("matplotlib not available. Skipping latency chart.")
            return ""
