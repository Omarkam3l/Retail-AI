import pytest
import os
import json
import tempfile
from src.evaluation.types import (
    BBoxAnnotation, GroundTruthAnnotation, PredictionRecord,
    AnnotationFormat, DatasetSplit, FailureCategory
)
from src.evaluation.config import EvaluationConfig, DatasetConfig
from src.evaluation.dataset_manager import DatasetManager
from src.evaluation.annotation_loader import AnnotationLoader
from src.evaluation.converters import DatasetConverter
from src.evaluation.prediction_logger import PredictionLogger
from src.evaluation.replay_runner import ReplayEvaluationRunner
from src.evaluation.metrics.detection import DetectionMetricsCalculator, compute_iou
from src.evaluation.metrics.tracking import TrackingMetricsCalculator
from src.evaluation.metrics.behavior import BehaviorMetricsCalculator
from src.evaluation.metrics.risk import RiskMetricsCalculator
from src.evaluation.metrics.alerts import AlertMetricsCalculator
from src.evaluation.metrics.overall import OverallMetricsCalculator
from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.report_generator import ReportGenerator


# --- Fixtures ---

def _make_gt(frame_index=0, bboxes=None, behaviors=None, risk_level=None, alerts=None, metadata=None):
    return GroundTruthAnnotation(
        frame_index=frame_index, timestamp_ms=frame_index * 33.33,
        image_path=f"frame_{frame_index}.jpg",
        bboxes=bboxes or [],
        behaviors=behaviors or [],
        risk_level=risk_level,
        alerts=alerts or [],
        metadata=metadata or {}
    )

def _make_pred(frame_index=0, detections=None, track_ids=None, behaviors=None,
               risk_level=None, alerts=None):
    return PredictionRecord(
        frame_index=frame_index, timestamp_ms=frame_index * 33.33,
        camera_id="cam_test",
        detections=detections or [],
        track_ids=track_ids or [],
        behaviors=behaviors or [],
        risk_level=risk_level,
        alerts=alerts or []
    )

def _box(x1, y1, x2, y2, label="person", conf=0.9, track_id=None):
    return BBoxAnnotation(x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                          class_label=label, confidence=conf, track_id=track_id)


# --- Dataset Manager ---

def test_dataset_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create split directories
        os.makedirs(os.path.join(tmpdir, "train"))
        os.makedirs(os.path.join(tmpdir, "test"))
        # Create dummy images
        for i in range(3):
            open(os.path.join(tmpdir, "train", f"img_{i}.jpg"), "w").close()
        open(os.path.join(tmpdir, "test", "img_0.jpg"), "w").close()

        dm = DatasetManager()
        meta = dm.register_dataset("test_ds", tmpdir, "1.0")
        assert meta.name == "test_ds"
        assert meta.splits["train"] == 3
        assert meta.splits["test"] == 1
        assert "test_ds" in dm.list_datasets()


# --- Annotation Loader ---

def test_annotation_loader_coco():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        coco_data = {
            "images": [{"id": 1, "file_name": "img1.jpg"}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]}]
        }
        json.dump(coco_data, f)
        f.flush()
        fpath = f.name

    try:
        loader = AnnotationLoader()
        annotations = loader.load(fpath, AnnotationFormat.COCO)
        assert len(annotations) == 1
        assert len(annotations[0].bboxes) == 1
        assert annotations[0].bboxes[0].class_label == "person"
        assert annotations[0].bboxes[0].x_min == 10
        assert annotations[0].bboxes[0].x_max == 60  # 10 + 50
    finally:
        os.unlink(fpath)


def test_annotation_loader_yolo():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a YOLO label file
        with open(os.path.join(tmpdir, "img1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.4 0.6\n")

        loader = AnnotationLoader()
        annotations = loader.load(tmpdir, AnnotationFormat.YOLO, class_names={0: "person"})
        assert len(annotations) == 1
        assert len(annotations[0].bboxes) == 1
        assert annotations[0].bboxes[0].class_label == "person"


def test_annotation_validation():
    loader = AnnotationLoader()
    annotations = [
        _make_gt(0, [_box(10, 10, 5, 5, "")]),  # invalid bbox + missing label
        _make_gt(0, [_box(0, 0, 10, 10, "person")])  # duplicate frame_index
    ]
    issues = loader.validate(annotations)
    assert issues["missing_labels"] == 1
    assert issues["invalid_bboxes"] == 1
    assert issues["duplicate_ids"] == 1


# --- Converters ---

def test_converter_coco_to_yolo():
    with tempfile.TemporaryDirectory() as tmpdir:
        annotations = [_make_gt(0, [_box(10, 20, 60, 80, "person")])]
        converter = DatasetConverter()
        count = converter.coco_to_yolo(annotations, {"person": 0}, tmpdir, img_width=100, img_height=100)
        assert count == 1
        files = os.listdir(tmpdir)
        assert len(files) == 1


# --- Detection Metrics ---

def test_detection_iou():
    a = _box(0, 0, 10, 10)
    b = _box(5, 5, 15, 15)
    iou = compute_iou(a, b)
    assert 0.0 < iou < 1.0  # Overlapping boxes

    c = _box(0, 0, 10, 10)
    d = _box(0, 0, 10, 10)
    assert compute_iou(c, d) == pytest.approx(1.0)


def test_detection_metrics():
    gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")])]
    pred = [_make_pred(0, [_box(0, 0, 10, 10, "person", conf=0.9)])]
    calc = DetectionMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


# --- Tracking Metrics ---

def test_tracking_metrics():
    gt = [
        _make_gt(0, [_box(0, 0, 10, 10, "person", track_id=1)]),
        _make_gt(1, [_box(1, 1, 11, 11, "person", track_id=1)])
    ]
    pred = [
        _make_pred(0, [_box(0, 0, 10, 10, "person")], track_ids=[1]),
        _make_pred(1, [_box(1, 1, 11, 11, "person")], track_ids=[1])
    ]
    calc = TrackingMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.mota > 0.0
    assert result.motp > 0.0


# --- Behavior Metrics ---

def test_behavior_metrics():
    gt = [_make_gt(0, behaviors=["loitering"]), _make_gt(1, behaviors=["loitering"])]
    pred = [_make_pred(0, behaviors=["loitering"]), _make_pred(1, behaviors=[])]
    calc = BehaviorMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.per_behavior["loitering"]["recall"] == pytest.approx(0.5)


# --- Risk Metrics ---

def test_risk_metrics():
    gt = [_make_gt(0, risk_level="HIGH"), _make_gt(1, risk_level="HIGH")]
    pred = [_make_pred(0, risk_level="HIGH"), _make_pred(1, risk_level="MEDIUM")]
    calc = RiskMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.precision > 0.0
    assert result.recall > 0.0


# --- Alert Metrics ---

def test_alert_metrics():
    gt = [_make_gt(0, alerts=["theft"]), _make_gt(1, alerts=["theft"])]
    pred = [_make_pred(0, alerts=["theft"]), _make_pred(1, alerts=[])]
    calc = AlertMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.5)
    assert result.missed_alerts == 1


# --- Overall Metrics ---

def test_overall_metrics():
    gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")], behaviors=["loitering"])]
    pred = [_make_pred(0, [_box(0, 0, 10, 10, "person")], behaviors=["loitering"])]
    calc = OverallMetricsCalculator()
    result = calc.compute(gt, pred)
    assert result.detection is not None
    assert result.tracking is not None
    assert result.behavior is not None


# --- Confusion Matrix ---

def test_confusion_matrix():
    gt = [_make_gt(0, behaviors=["loitering"]), _make_gt(1, behaviors=[])]
    pred = [_make_pred(0, behaviors=["loitering"]), _make_pred(1, behaviors=["loitering"])]
    cm = ConfusionMatrixGenerator(categories=["loitering"])
    matrix = cm.generate(gt, pred)
    assert matrix["loitering"]["loitering"] >= 1
    md = cm.to_markdown(matrix)
    assert "loitering" in md


# --- Error Analysis ---

def test_error_analysis():
    gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")], metadata={"conditions": "dark"})]
    pred = [_make_pred(0, [])]  # Missing detection
    analyzer = ErrorAnalyzer()
    failures = analyzer.analyze(gt, pred)
    assert failures["lighting"] >= 1


# --- Threshold Optimizer ---

def test_threshold_optimizer():
    gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")])]
    pred = [_make_pred(0, [_box(0, 0, 10, 10, "person", conf=0.5)])]
    opt = ThresholdOptimizer(detection_confidences=[0.3, 0.5, 0.7])
    result = opt.optimize(gt, pred)
    assert "best" in result
    assert "all_results" in result


# --- Experiment Tracker ---

def test_experiment_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(storage_dir=tmpdir)
        exp = tracker.create_experiment(
            git_commit="abc123", dataset_version="1.0", model_version="yolo11s"
        )
        assert exp.experiment_id
        path = tracker.save(exp.experiment_id)
        assert os.path.isfile(path)
        loaded = tracker.load(exp.experiment_id)
        assert loaded.git_commit == "abc123"


# --- Benchmark Runner ---

def test_benchmark_runner():
    gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")])]
    pred = [_make_pred(0, [_box(0, 0, 10, 10, "person")])]
    runner = BenchmarkRunner()
    result = runner.run_benchmark("test_bench", gt, pred, model_version="v1")
    table = runner.get_comparison_table()
    assert len(table) == 1
    assert table[0]["name"] == "test_bench"


# --- Report Generator ---

def test_report_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator(output_dir=tmpdir)
        gt = [_make_gt(0, [_box(0, 0, 10, 10, "person")])]
        pred = [_make_pred(0, [_box(0, 0, 10, 10, "person")])]
        calc = OverallMetricsCalculator()
        result = calc.compute(gt, pred)
        report = gen.generate_markdown(result, title="Test Report")
        assert "Detection Metrics" in report
        assert "Tracking Metrics" in report


# --- Prediction Logger ---

def test_prediction_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        pl = PredictionLogger(output_dir=tmpdir)
        record = _make_pred(0, [_box(0, 0, 10, 10, "person")])
        pl.log(record)
        path = pl.flush("test_run")
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1


# --- Replay Runner ---

def test_replay_runner():
    gt_list = [_make_gt(0), _make_gt(1)]
    def mock_pipeline(gt: GroundTruthAnnotation) -> PredictionRecord:
        return PredictionRecord(
            frame_index=gt.frame_index, timestamp_ms=gt.timestamp_ms,
            camera_id="cam_test"
        )
    runner = ReplayEvaluationRunner()
    preds = runner.run(gt_list, mock_pipeline)
    assert len(preds) == 2


# --- Config Validation ---

def test_config_validation():
    config = EvaluationConfig(
        datasets=[DatasetConfig(name="test", root_path="/tmp/test")]
    )
    assert config.metrics.compute_detection is True
    assert len(config.thresholds.detection_confidences) == 5
