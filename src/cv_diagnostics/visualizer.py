"""Handles rendering of analysis visual charts: NMS comparison, histograms, and heatmaps."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.cv_diagnostics.types import FrameRecord

class DiagnosticsVisualizer:
    """Generates JPEG/PNG charts of metric distributions and spatial density heatmaps."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    def generate_confidence_histogram(self, confidences: List[float], class_name: str) -> str:
        plt.figure()
        plt.hist(confidences, bins=20, alpha=0.75, color='blue', edgecolor='black')
        plt.title(f"Confidence Distribution - {class_name}")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        
        path = os.path.join(self._output_dir, "histograms", f"conf_{class_name.lower()}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=100)
        plt.close()
        return path

    def generate_spatial_heatmap(self, coordinates: List[Tuple[float, float]], filename: str) -> str:
        # Create blank overlay
        heatmap = np.zeros((480, 640), dtype=np.float32)
        for x, y in coordinates:
            cx, cy = int(x * 640), int(y * 480)
            if 0 <= cx < 640 and 0 <= cy < 480:
                heatmap[cy, cx] += 1.0

        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        heatmap_img = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        path = os.path.join(self._output_dir, "heatmaps", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, colored)
        return path
