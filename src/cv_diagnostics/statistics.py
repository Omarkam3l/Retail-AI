"""Computes precise tracking, detection, NMS, and stability statistics."""
import numpy as np
from typing import Dict, List, Any
from src.cv_diagnostics.types import FrameRecord, FailureRecord

class StatisticsCalculator:
    """Performs metric calculations over recorded frame arrays."""

    @staticmethod
    def calculate_jitter(track_history: List[Any]) -> Dict[str, float]:
        if len(track_history) < 2:
            return {"center_var": 0.0, "w_var": 0.0, "h_var": 0.0, "iou_mean": 1.0}
        
        centers = [b.center for b in track_history]
        widths = [b.width for b in track_history]
        heights = [b.height for b in track_history]
        
        # Consec IoU
        ious = []
        for i in range(1, len(track_history)):
            box1 = track_history[i-1]
            box2 = track_history[i]
            # Simple IoU calculator
            x_left = max(box1.x_min, box2.x_min)
            y_top = max(box1.y_min, box2.y_min)
            x_right = min(box1.x_max, box2.x_max)
            y_bottom = min(box1.y_max, box2.y_max)
            if x_right < x_left or y_bottom < y_top:
                ious.append(0.0)
            else:
                inter = (x_right - x_left) * (y_bottom - y_top)
                union = box1.width * box1.height + box2.width * box2.height - inter
                ious.append(inter / union if union > 0.0 else 0.0)
                
        return {
            "center_var": float(np.var(centers, axis=0).sum()),
            "w_var": float(np.var(widths)),
            "h_var": float(np.var(heights)),
            "iou_mean": float(np.mean(ious))
        }

    @staticmethod
    def compile_report_stats(records: List[FrameRecord]) -> Dict[str, Any]:
        stats = {
            "total_frames": len(records),
            "total_detections": sum(len(r.detections_after_nms) for r in records),
            "total_failures": sum(len(r.failures) for r in records),
            "id_switches": 0,
            "track_lifetimes": {},
            "class_metrics": {},
            "small_objects_buckets": {"<16x16": 0, "16-32": 0, "32-64": 0, "64-128": 0, "128+": 0}
        }
        
        all_track_ids = set()
        for r in records:
            for t in r.tracks:
                all_track_ids.add(t.track_id)
                if t.track_id not in stats["track_lifetimes"]:
                    stats["track_lifetimes"][t.track_id] = 0
                stats["track_lifetimes"][t.track_id] += 1
                
                # Class mapping
                cls = t.class_label
                if cls not in stats["class_metrics"]:
                    stats["class_metrics"][cls] = {"detections": 0, "sum_conf": 0.0}
                stats["class_metrics"][cls]["detections"] += 1
                stats["class_metrics"][cls]["sum_conf"] += t.confidence
                
                # Bbox size buckets (assuming 640x640 norm area)
                area = (t.bbox.width * 640) * (t.bbox.height * 640)
                side = np.sqrt(area)
                if side < 16:
                    stats["small_objects_buckets"]["<16x16"] += 1
                elif side < 32:
                    stats["small_objects_buckets"]["16-32"] += 1
                elif side < 64:
                    stats["small_objects_buckets"]["32-64"] += 1
                elif side < 128:
                    stats["small_objects_buckets"]["64-128"] += 1
                else:
                    stats["small_objects_buckets"]["128+"] += 1

        stats["total_tracks"] = len(all_track_ids)
        return stats
