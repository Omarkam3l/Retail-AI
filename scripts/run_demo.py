"""
Retail AI Surveillance Platform — Live Video Demo
===================================================
Usage:
    python scripts/run_demo.py --video path/to/video.mp4
    python scripts/run_demo.py --video 0              # webcam
    python scripts/run_demo.py --video path/to/video.mp4 --save output.mp4
"""
import argparse
import sys
import os
import time
import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.yolo_detector import YOLO11Detector
from src.tracking.adapter import ByteTrackAdapter
from src.association.engine import ObjectAssociationEngine
from src.behavior.engine import BehaviorEngine
from src.behavior.rules import PocketConcealmentRule, LoiteringRule
from src.risk.engine import RiskAssessmentEngine
from src.alerts.engine import AlertEvidenceEngine
from src.alerts.dispatcher import MockNotificationDispatcher
from src.inference.orchestrator import PipelineOrchestrator
from src.inference.event_bus import EventBus
from src.common.types import ClassLabel


# ──────────────────────────────── Drawing Helpers ────────────────────────────────

from src.ingestion.utils import draw_rounded_rect, draw_rounded_rect_filled

COLORS = {
    ClassLabel.PERSON: (180, 119, 31),        # Color-blind friendly Blue
    ClassLabel.BACKPACK: (14, 127, 255),      # Color-blind friendly Orange
    ClassLabel.HANDBAG: (44, 160, 44),        # Color-blind friendly Green
    ClassLabel.SHOPPING_BASKET: (214, 39, 40), # Color-blind friendly Red
    ClassLabel.SHOPPING_CART: (214, 39, 40),
    ClassLabel.SHELF_ITEM: (188, 189, 34),    # Color-blind friendly Olive
}

# Cache for track trails
TRAILS_CACHE = {}

def draw_detections(frame, metadata):
    """Draw tracked persons, objects, and their enriched metadata on the frame."""
    h, w = frame.shape[:2]

    # Draw persons
    for person in metadata.persons:
        x1 = int(person.bbox.x_min * w)
        y1 = int(person.bbox.y_min * h)
        x2 = int(person.bbox.x_max * w)
        y2 = int(person.bbox.y_max * h)

        # Draw rounded bounding box
        draw_rounded_rect(frame, (x1, y1), (x2, y2), (180, 119, 31), thickness=3, radius=12)
        
        # Bounding box confidence bar
        bar_h = int((y2 - y1) * person.confidence)
        cv2.rectangle(frame, (x1 - 6, y2 - bar_h), (x1 - 2, y2), (180, 119, 31), -1)

        # Track trails logic
        tid = person.track_id
        if tid not in TRAILS_CACHE:
            TRAILS_CACHE[tid] = []
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        TRAILS_CACHE[tid].append(center)
        if len(TRAILS_CACHE[tid]) > 15:
            TRAILS_CACHE[tid].pop(0)

        # Draw trails
        for i in range(1, len(TRAILS_CACHE[tid])):
            cv2.line(frame, TRAILS_CACHE[tid][i-1], TRAILS_CACHE[tid][i], (180, 119, 31), 2, lineType=cv2.LINE_AA)
        
        # Build text lines for person metadata
        lines = [
            f"PERSON ID: {person.track_id} ({person.confidence:.2f})",
            f"Risk Score: {getattr(person, 'risk_score', 0.0):.1f}"
        ]
        
        alert_lvl = getattr(person, 'alert_level', None)
        if alert_lvl:
            lines.append(f"Alert: {alert_lvl}")
            
        flags = getattr(person, 'behavior_flags', [])
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
            
        # Draw background rectangle for person text banner
        y_text = y1 - 8
        for line in reversed(lines):
            (tw, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # Rounded filled label background
            draw_rounded_rect_filled(frame, (x1, y_text - th - 6), (x1 + tw + 8, y_text + baseline + 2), (180, 119, 31), radius=4)
            
            # Color coding for alert
            color = (255, 255, 255)
            cv2.putText(frame, line, (x1 + 4, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)
            y_text -= (th + 6)

    # Draw objects
    for obj in metadata.objects:
        color = COLORS.get(obj.class_label, (200, 200, 200))
        x1 = int(obj.bbox.x_min * w)
        y1 = int(obj.bbox.y_min * h)
        x2 = int(obj.bbox.x_max * w)
        y2 = int(obj.bbox.y_max * h)

        # Draw rounded bounding box
        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2, radius=8)
        
        # Bounding box confidence bar
        bar_h = int((y2 - y1) * obj.confidence)
        cv2.rectangle(frame, (x1 - 6, y2 - bar_h), (x1 - 2, y2), color, -1)
        
        lines = [
            f"{obj.class_label.value.upper()} ID: {obj.track_id} ({obj.confidence:.2f})"
        ]
        if obj.sku:
            lines.append(f"SKU: {obj.sku}")
            lines.append(f"Brand: {obj.brand} ({obj.category})")
            lines.append(f"Sim: {obj.similarity:.2f} | RecConf: {obj.rec_confidence:.2f}")
        else:
            lines.append("Product: UNKNOWN")
            
        # Draw background rectangle for object text banner
        y_text = y1 - 8
        for line in reversed(lines):
            (tw, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            draw_rounded_rect_filled(frame, (x1, y_text - th - 6), (x1 + tw + 8, y_text + baseline + 2), color, radius=4)
            cv2.putText(frame, line, (x1 + 4, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            y_text -= (th + 6)


def draw_alerts(frame, alerts):
    """Draw alert banners on the frame."""
    for i, alert in enumerate(alerts):
        text = f"ALERT: {alert.level.value} | Track {alert.track_id} | {alert.event_type}"
        y_pos = 30 + i * 30
        cv2.putText(frame, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_hud(frame, fps, frame_idx, num_persons, num_objects, profiler):
    """Draw a heads-up display with live statistics."""
    h, w = frame.shape[:2]

    # Semi-transparent dark banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 280, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    x_off = w - 270
    cv2.putText(frame, f"FPS: {fps:.1f}", (x_off, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_idx}", (x_off, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Persons: {num_persons}", (x_off, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"Objects: {num_objects}", (x_off, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # Stage latencies
    avg = profiler.get_summary()
    y = 125
    for stage in ["detection", "tracking", "association", "behavior"]:
        if stage in avg:
            cv2.putText(frame, f"{stage[:3]}: {avg[stage]:.1f}ms", (x_off, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y += 18


# ──────────────────────────────── Main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retail AI Surveillance — Live Demo")
    parser.add_argument("--video", required=True, help="Path to video file or webcam index (0)")
    parser.add_argument("--save", default=None, help="Path to save output video (e.g. output.mp4)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--no-display", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    # ── Build Pipeline Components ──
    print("=" * 60)
    print("  Retail AI Surveillance Platform — Live Demo")
    print("=" * 60)

    print("[1/6] Loading YOLO11s detector...")
    detector = YOLO11Detector(
        model_path="yolo11s.pt",
        confidence_threshold=args.conf,
        device=args.device
    )

    print("[2/6] Initializing ByteTrack tracker...")
    tracker = ByteTrackAdapter(track_threshold=0.25)

    print("[3/6] Setting up association engine...")
    association = ObjectAssociationEngine(proximity_threshold=0.3, persistence_threshold=3)

    print("[4/6] Loading behavior rules...")
    behavior = BehaviorEngine()
    behavior.register_rule(PocketConcealmentRule())
    behavior.register_rule(LoiteringRule(loiter_threshold_seconds=20.0))

    print("[5/6] Configuring risk & alert engines...")
    risk_engine = RiskAssessmentEngine()
    dispatcher = MockNotificationDispatcher()
    alert_engine = AlertEvidenceEngine(dispatcher=dispatcher)

    # Product Recognition Engine Integration
    print("[5.5/6] Loading product recognition model and catalog...")
    try:
        from src.product_recognition.embedding_model import DINOv2EmbeddingModel
        from src.product_recognition.feature_extractor import FeatureExtractor
        from src.product_recognition.catalog_loader import CatalogLoader
        from src.product_recognition.similarity import SimilarityEngine
        from src.product_recognition.matcher import ProductMatcher
        from src.product_recognition.recognition_engine import ProductRecognitionEngine
        
        emb_model = DINOv2EmbeddingModel(model_name="dinov2_vits14", device=args.device)
        extractor = FeatureExtractor(emb_model)
        
        # Load catalog and fail fast if missing/empty
        catalog = CatalogLoader.load_from_json("data/product_catalog.json")
        if not catalog.list_all():
            raise ValueError("Catalog is empty.")
            
        sim_engine = SimilarityEngine(catalog)
        matcher = ProductMatcher(sim_engine)
        recognition_engine = ProductRecognitionEngine(extractor, matcher)
        print("  ✅ Product Recognition Engine and Catalog loaded successfully.")
    except Exception as e:
        print(f"  ❌ FATAL ERROR: Failed to load Product Catalog: {e}")
        sys.exit(1)

    print("[6/6] Assembling pipeline orchestrator...")
    orchestrator = PipelineOrchestrator(
        camera_id="demo_cam",
        detector=detector,
        tracker=tracker,
        association_engine=association,
        behavior_engine=behavior,
        risk_engine=risk_engine,
        alert_engine=alert_engine,
        recognition_engine=recognition_engine
    )
    orchestrator.initialize()
    print("Pipeline ready!\n")

    # ── Open Video Source ──
    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {args.video}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps_video:.1f} FPS | {total_frames} frames")

    # ── Output Writer ──
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_video, (width, height))
        print(f"Saving output to: {args.save}")

    # ── Processing Loop ──
    frame_idx = 0
    total_alerts = 0
    fps_counter = 0
    fps_start = time.perf_counter()
    running_fps = 0.0

    print("\nProcessing... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = frame_idx * (1000.0 / fps_video)

        # Run full pipeline
        metadata, alerts = orchestrator.process_frame(frame, frame_idx, timestamp_ms)
        total_alerts += len(alerts)

        # Draw visualizations
        draw_detections(frame, metadata)
        draw_alerts(frame, alerts)

        # FPS calculation
        fps_counter += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            running_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.perf_counter()

        draw_hud(frame, running_fps, frame_idx,
                 len(metadata.persons), len(metadata.objects),
                 orchestrator.profiler)

        # Save frame
        if writer:
            writer.write(frame)

        # Display frame
        if not args.no_display:
            cv2.imshow("Retail AI Surveillance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nStopping (user pressed 'q')...")
                break

        frame_idx += 1

        # Progress logging every 100 frames
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | "
                  f"FPS: {running_fps:.1f} | "
                  f"Persons: {len(metadata.persons)} | "
                  f"Alerts: {total_alerts}")

    # ── Cleanup ──
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    orchestrator.shutdown()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Run Complete")
    print("=" * 60)
    print(f"  Frames Processed: {frame_idx}")
    print(f"  Total Alerts:     {total_alerts}")

    avg = orchestrator.profiler.get_summary()
    if avg:
        print(f"\n  Average Stage Latencies:")
        for stage, ms in avg.items():
            print(f"    {stage:>15s}: {ms:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
