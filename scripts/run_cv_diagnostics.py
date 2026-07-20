#!/usr/bin/env python3
"""
Retail CV Diagnostics CLI Runner
===============================
Runs the full Computer Vision diagnostics cascade on a video stream,
classifies pipeline failures, tracks lifetimes, generates visualization
charts/heatmaps, and compiles a comprehensive markdown report.
"""
import argparse
import sys
import os
import cv2
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cv_diagnostics.types import FrameRecord, DetectionRecord, TrackRecord, RecognitionRecord, AssociationRecord, FailureRecord
from src.cv_diagnostics.recorder import DiagnosticsRecorder
from src.cv_diagnostics.statistics import StatisticsCalculator
from src.cv_diagnostics.visualizer import DiagnosticsVisualizer
from src.cv_diagnostics.reporter import DiagnosticsReporter

from src.detection.yolo_detector import YOLO11Detector
from src.tracking.adapter import ByteTrackAdapter
from src.association.engine import ObjectAssociationEngine
from src.behavior.engine import BehaviorEngine
from src.behavior.rules import PocketConcealmentRule, LoiteringRule
from src.risk.engine import RiskAssessmentEngine
from src.alerts.engine import AlertEvidenceEngine
from src.inference.orchestrator import PipelineOrchestrator
from src.common.types import BoundingBox, DetectedObject, TrackedPerson, FrameMetadata


class TrackTimelineLogger:
    """Logs chronological events for each track ID across frames."""
    def __init__(self) -> None:
        self.timeline = {}

    def log_event(self, track_id: int, event_str: str) -> None:
        if track_id not in self.timeline:
            self.timeline[track_id] = []
        self.timeline[track_id].append(event_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail CV Diagnostics Suite CLI Runner")
    parser.add_argument(
        "--video",
        type=str,
        default="data/shoplifting2.mp4",
        help="Path to input video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/diagnostics_run",
        help="Directory to save generated visual charts, cropped failures, and report.md."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detector."
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for YOLO detector."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: 'cpu', 'cuda', or 'auto'."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit execution to a maximum number of video frames."
    )
    return parser.parse_args()


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates intersection-over-union between two bounding boxes."""
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    box2_area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0.0 else 0.0


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "failures"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "heatmaps"), exist_ok=True)

    print("=== Initializing Retail Diagnostics Suite ===")
    print(f"Input Video: {args.video}")
    print(f"Output Directory: {args.output}")
    print(f"Device: {args.device} | Confidence: {args.conf} | IoU: {args.nms_iou}")

    # 1. Initialize Pipeline Components
    detector = YOLO11Detector(
        model_path="yolo11s.pt",
        confidence_threshold=args.conf,
        nms_iou_threshold=args.nms_iou,
        device=args.device
    )
    tracker = ByteTrackAdapter(track_threshold=args.conf)
    association = ObjectAssociationEngine(proximity_threshold=0.25)
    
    behavior = BehaviorEngine()
    behavior.register_rule(PocketConcealmentRule())
    behavior.register_rule(LoiteringRule(loiter_threshold_seconds=5.0))
    
    risk_engine = RiskAssessmentEngine()
    alert_engine = AlertEvidenceEngine()

    orchestrator = PipelineOrchestrator(
        camera_id="diagnostics_cam",
        detector=detector,
        tracker=tracker,
        association_engine=association,
        behavior_engine=behavior,
        risk_engine=risk_engine,
        alert_engine=alert_engine
    )
    orchestrator.initialize()

    # 2. Hook / Intercept Stages to capture raw telemetry
    last_detections = []
    original_detect = detector.detect
    def wrapped_detect(frame_np):
        nonlocal last_detections
        last_detections = original_detect(frame_np)
        return last_detections
    detector.detect = wrapped_detect

    last_associations = {}
    original_associate = association.associate
    def wrapped_associate(*a_args, **a_kwargs):
        nonlocal last_associations
        last_associations = original_associate(*a_args, **a_kwargs)
        return last_associations
    association.associate = wrapped_associate

    # Initialize Diagnostics tools
    recorder = DiagnosticsRecorder()
    timeline_logger = TrackTimelineLogger()
    track_history = {}  # Key: track_id -> List[BoundingBox]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    print("=== Processing Video Stream ===")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = frame_idx * (1000.0 / fps)
        
        # Run orchestrator processing cascade
        metadata, alerts = orchestrator.process_frame(frame, frame_idx, timestamp_ms)

        # Retrieve latencies from orchestrator's profiler
        latency_ms = {}
        with orchestrator.profiler._lock:
            for stage, q in orchestrator.profiler._stage_latencies.items():
                if q:
                    latency_ms[stage] = q[-1]

        # Extract pre-NMS candidate detections using lower confidence and high IoU
        raw_candidates = []
        if detector._model is not None:
            try:
                letterboxed_img, ratio, pad = detector.letterbox(frame, new_shape=640)
                results_pre = detector._model.predict(
                    letterboxed_img,
                    device=detector._device,
                    conf=0.1,  # Capture low-confidence raw boxes
                    iou=0.9,   # Keep overlap candidates
                    half=detector._fp16 and detector._device == "cuda",
                    verbose=False
                )
                raw_candidates = detector._postprocess(results_pre, frame.shape, is_letterbox=True, ratio=ratio, pad=pad)
            except Exception as e:
                print(f"Warning: Failed to fetch pre-NMS candidates at frame {frame_idx}: {e}")

        # Compute NMS suppressed records
        detections_before_nms = []
        for cand in raw_candidates:
            is_matched = False
            for det in last_detections:
                if det.class_label == cand.class_label:
                    if calculate_iou(det.bbox, cand.bbox) > 0.8:
                        is_matched = True
                        break

            is_nms_suppressed = False
            nms_suppression_reason = ""
            if not is_matched:
                for det in last_detections:
                    if det.class_label == cand.class_label and det.confidence > cand.confidence:
                        iou_val = calculate_iou(det.bbox, cand.bbox)
                        if iou_val > args.nms_iou:
                            is_nms_suppressed = True
                            nms_suppression_reason = f"Suppressed by NMS: IoU={iou_val:.2f} with detection (conf: {det.confidence:.2f})"
                            break

            detections_before_nms.append(DetectionRecord(
                bbox=cand.bbox,
                confidence=cand.confidence,
                class_label=cand.class_label.value if hasattr(cand.class_label, 'value') else str(cand.class_label),
                is_nms_suppressed=is_nms_suppressed,
                nms_suppression_reason=nms_suppression_reason
            ))

        detections_after_nms = [
            DetectionRecord(
                bbox=det.bbox,
                confidence=det.confidence,
                class_label=det.class_label.value if hasattr(det.class_label, 'value') else str(det.class_label),
                is_nms_suppressed=False
            )
            for det in last_detections
        ]

        # Extract track records
        tracks = []
        for p in metadata.persons:
            tracks.append(TrackRecord(
                track_id=p.track_id,
                class_label="person",
                bbox=p.bbox,
                confidence=p.confidence,
                velocity=p.velocity
            ))
        for o in metadata.objects:
            tracks.append(TrackRecord(
                track_id=o.track_id if o.track_id is not None else -1,
                class_label=o.class_label.value if hasattr(o.class_label, 'value') else str(o.class_label),
                bbox=o.bbox,
                confidence=o.confidence,
                velocity=(0.0, 0.0)
            ))

        # Extract recognition records
        recognitions = []
        for o in metadata.objects:
            if o.track_id is not None and (o.sku or o.brand or o.category):
                recognitions.append(RecognitionRecord(
                    track_id=o.track_id,
                    sku=o.sku,
                    brand=o.brand,
                    category=o.category,
                    similarity=o.similarity or 0.0,
                    confidence=o.rec_confidence or 0.0
                ))

        # Extract association records
        associations = []
        for p_id, obj_dict in last_associations.items():
            for o_id, state in obj_dict.items():
                associations.append(AssociationRecord(
                    person_track_id=p_id,
                    object_track_id=o_id,
                    confidence=1.0,
                    state=state.value if hasattr(state, 'value') else str(state)
                ))

        # Classify Failure Records
        failures = []

        # 1. Low Confidence Detection Failure
        for det in last_detections:
            if det.confidence < 0.45:
                label_str = det.class_label.value if hasattr(det.class_label, 'value') else str(det.class_label)
                crop_filename = f"crop_f{frame_idx}_low_conf_{label_str}.jpg"
                failures.append(FailureRecord(
                    frame_index=frame_idx,
                    track_id=det.track_id,
                    class_label=label_str,
                    category="low_confidence",
                    reason=f"Detection confidence {det.confidence:.2f} is below target 0.45 threshold.",
                    recommendation="Review area illumination or check detector training parameters.",
                    confidence=det.confidence,
                    bbox=det.bbox,
                    crop_filename=crop_filename
                ))

        # 2. Bounding Box Jitter Failure
        for t in tracks:
            tid = t.track_id
            if tid == -1:
                continue
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append(t.bbox)

            jitter_metrics = StatisticsCalculator.calculate_jitter(track_history[tid])
            if len(track_history[tid]) >= 3 and jitter_metrics["iou_mean"] < 0.65:
                crop_filename = f"crop_f{frame_idx}_track_{tid}_jitter.jpg"
                failures.append(FailureRecord(
                    frame_index=frame_idx,
                    track_id=tid,
                    class_label=t.class_label,
                    category="track_jitter",
                    reason=f"Track bounding box jitter detected (Average consecutive IoU: {jitter_metrics['iou_mean']:.2f}).",
                    recommendation="Enable Kalman filtering box smoothing or verify feature matcher weights.",
                    confidence=t.confidence,
                    bbox=t.bbox,
                    crop_filename=crop_filename
                ))

        # 3. Small Object Failures
        for det in last_detections:
            area = det.bbox.width * det.bbox.height
            if area < 0.0015:
                label_str = det.class_label.value if hasattr(det.class_label, 'value') else str(det.class_label)
                crop_filename = f"crop_f{frame_idx}_small_{label_str}.jpg"
                failures.append(FailureRecord(
                    frame_index=frame_idx,
                    track_id=det.track_id,
                    class_label=label_str,
                    category="small_object",
                    reason=f"Object area too small ({area:.6f} normalized area) for robust tracking.",
                    recommendation="Position camera closer to high-density shelf areas or increase source resolution.",
                    confidence=det.confidence,
                    bbox=det.bbox,
                    crop_filename=crop_filename
                ))

        # 4. NMS Suppression Failures
        for record in detections_before_nms:
            if record.is_nms_suppressed:
                crop_filename = f"crop_f{frame_idx}_suppressed_{record.class_label}.jpg"
                failures.append(FailureRecord(
                    frame_index=frame_idx,
                    track_id=None,
                    class_label=record.class_label,
                    category="nms_suppression",
                    reason=record.nms_suppression_reason,
                    recommendation="Adjust YOLO detector NMS IoU settings or apply soft-NMS.",
                    confidence=record.confidence,
                    bbox=record.bbox,
                    crop_filename=crop_filename
                ))

        # Save failure crops
        if failures:
            h, w = frame.shape[:2]
            for fail in failures:
                x_min = int(fail.bbox.x_min * w)
                y_min = int(fail.bbox.y_min * h)
                x_max = int(fail.bbox.x_max * w)
                y_max = int(fail.bbox.y_max * h)

                x_min = max(0, min(w - 1, x_min))
                y_min = max(0, min(h - 1, y_min))
                x_max = max(0, min(w - 1, x_max))
                y_max = max(0, min(h - 1, y_max))

                if x_max > x_min and y_max > y_min:
                    crop = frame[y_min:y_max, x_min:x_max]
                    cv2.imwrite(os.path.join(args.output, "failures", fail.crop_filename), crop)

        # Build FrameRecord and add to recorder
        record = FrameRecord(
            frame_index=frame_idx,
            timestamp_ms=timestamp_ms,
            raw_frame=frame,
            letterboxed_frame=detector.letterbox(frame, new_shape=640)[0],
            detections_before_nms=detections_before_nms,
            detections_after_nms=detections_after_nms,
            tracks=tracks,
            recognitions=recognitions,
            associations=associations,
            failures=failures,
            latency_ms=latency_ms
        )
        recorder.record_frame(record)

        # Timeline Logging
        # Persons
        for p in metadata.persons:
            tid = p.track_id
            if tid not in timeline_logger.timeline:
                timeline_logger.log_event(tid, f"Frame {frame_idx} ({timestamp_ms:.1f}ms): Track initialized for person (conf: {p.confidence:.2f})")
            
            # log behavior flags
            for flag in p.behavior_flags:
                timeline_logger.log_event(tid, f"Frame {frame_idx} ({timestamp_ms:.1f}ms): Behavior flag '{flag}' raised")

            # log risk
            if p.risk_score > 0.0:
                timeline_logger.log_event(tid, f"Frame {frame_idx} ({timestamp_ms:.1f}ms): Risk score updated to {p.risk_score:.2f}")

            # log alert level
            if p.alert_level:
                timeline_logger.log_event(tid, f"Frame {frame_idx} ({timestamp_ms:.1f}ms): Alert level '{p.alert_level}' triggered")

        # Objects / Failures
        for fail in failures:
            if fail.track_id is not None:
                timeline_logger.log_event(fail.track_id, f"Frame {frame_idx} ({timestamp_ms:.1f}ms): Pipeline failure '{fail.category}' - {fail.reason}")

        # Console feedback
        if frame_idx % 20 == 0:
            print(f"Processed frame {frame_idx} (Timestamp: {timestamp_ms:.1f}ms) | Failures: {len(failures)} | Tracks: {len(tracks)}")

        frame_idx += 1
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

    cap.release()
    orchestrator.shutdown()
    print("=== Processing Complete ===")

    # 3. Generate Visual Charts and Spatial Heatmaps
    print("=== Generating Diagnostic Visualization Charts ===")
    confidences_by_class = {}
    spatial_coords = []

    for r in recorder.get_all_records():
        for t in r.tracks:
            cls = t.class_label
            if cls not in confidences_by_class:
                confidences_by_class[cls] = []
            confidences_by_class[cls].append(t.confidence)
            spatial_coords.append(t.bbox.center)

    visualizer = DiagnosticsVisualizer(args.output)

    for cls_name, confs in confidences_by_class.items():
        if confs:
            try:
                hist_path = visualizer.generate_confidence_histogram(confs, cls_name)
                print(f"Generated confidence histogram for {cls_name} at: {hist_path}")
            except Exception as e:
                print(f"Error generating confidence histogram for {cls_name}: {e}")

    if spatial_coords:
        try:
            heatmap_path = visualizer.generate_spatial_heatmap(spatial_coords, "track_density.jpg")
            print(f"Generated spatial heatmap at: {heatmap_path}")
        except Exception as e:
            print(f"Error generating spatial heatmap: {e}")

    # 4. Compile Markdown Diagnostics Report
    print("=== Compiling Markdown Report ===")
    stats = StatisticsCalculator.compile_report_stats(recorder.get_all_records())
    
    all_failures = []
    for r in recorder.get_all_records():
        all_failures.extend(r.failures)

    reporter = DiagnosticsReporter(args.output)
    report_path = reporter.compile_report(stats, all_failures, timeline_logger.timeline)
    print(f"Report successfully compiled at: {report_path}")
    print("=== CV Diagnostics Complete ===")


if __name__ == "__main__":
    main()
