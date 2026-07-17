# Evaluation & Benchmarking Platform

Independent evaluation framework for the Retail AI Surveillance Platform.

## Architecture

- **Dataset Manager**: Version-aware dataset management with train/val/test splits
- **Annotation Loader**: COCO, YOLO, CVAT XML, Label Studio JSON parsers
- **Converters**: Bidirectional format converters (COCO ↔ YOLO, COCO ↔ CVAT)
- **Replay Runner**: Deterministic offline evaluation pipeline
- **Prediction Logger**: Structured JSON prediction logging
- **Metrics**: Detection (mAP), Tracking (MOTA/MOTP/IDF1), Behavior (P/R/F1), Risk, Alerts
- **Confusion Matrix**: Per-behavior normalized matrices
- **Error Analysis**: Automatic failure categorization
- **Threshold Optimizer**: Grid search for optimal configurations
- **Benchmark Runner**: Multi-dataset/model comparison
- **Experiment Tracker**: Versioned experiment storage
- **Report Generator**: Markdown/HTML/PDF reports
- **Visualization**: Charts, curves, and heatmaps

## Usage

```python
from src.evaluation.metrics.overall import OverallMetricsCalculator
from src.evaluation.annotation_loader import AnnotationLoader
from src.evaluation.types import AnnotationFormat

# Load ground truth
loader = AnnotationLoader()
gt = loader.load("annotations.json", AnnotationFormat.COCO)

# Compute metrics
calc = OverallMetricsCalculator()
result = calc.compute(gt, predictions)
```
