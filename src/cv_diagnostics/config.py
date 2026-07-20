"""Configuration module for custom CV diagnostics options."""
from dataclasses import dataclass

@dataclass
class DiagnosticsConfig:
    save_every_frame: bool = False
    save_only_failures: bool = True
    frame_interval: int = 1
    jpeg_quality: int = 85
    enable_heatmaps: bool = True
    enable_histograms: bool = True
    enable_failure_gallery: bool = True
    enable_stage_images: bool = True
    max_gallery_size: int = 100
    recovery_threshold: float = 0.82
