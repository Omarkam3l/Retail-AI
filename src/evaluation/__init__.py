from src.evaluation.types import (
    AnnotationFormat, DatasetSplit, FailureCategory,
    BBoxAnnotation, GroundTruthAnnotation, PredictionRecord,
    DetectionMetricsResult, TrackingMetricsResult, BehaviorMetricsResult,
    RiskMetricsResult, AlertMetricsResult, OverallEvaluationResult,
    ExperimentRecord, DatasetMetadata
)
from src.evaluation.config import EvaluationConfig
from src.evaluation.dataset_manager import DatasetManager
from src.evaluation.annotation_loader import AnnotationLoader
from src.evaluation.converters import DatasetConverter
from src.evaluation.prediction_logger import PredictionLogger
from src.evaluation.replay_runner import ReplayEvaluationRunner
from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.visualization import EvaluationVisualizer
