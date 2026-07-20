"""Compiles analysis markdown reports and track timeline logs from recorder logs."""
import os
from typing import List, Dict, Any

class DiagnosticsReporter:
    """Generates report.md detailing camera failures and track timeline histories."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    def compile_report(self, stats: Dict[str, Any], failures_list: List[Any], timeline_dict: Dict[int, List[str]]) -> str:
        report_path = os.path.join(self._output_dir, "report.md")
        os.makedirs(self._output_dir, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write("# Computer Vision Pipeline Diagnostics Report\n\n")
            f.write("## 1. Executive Summary\n")
            f.write(f"- **Total Frames Processed**: {stats['total_frames']}\n")
            f.write(f"- **Total Detections**: {stats['total_detections']}\n")
            f.write(f"- **Total Active Tracks**: {stats['total_tracks']}\n")
            f.write(f"- **Total Classified Failures**: {stats['total_failures']}\n\n")
            
            f.write("## 2. Failure Classification Summary\n")
            f.write("| Frame | Track ID | Class | Failure Category | Reason / Recommendation |\n")
            f.write("|---|---|---|---|---|\n")
            for fail in failures_list[:30]:  # Limit output rows
                f.write(f"| {fail.frame_index} | {fail.track_id} | {fail.class_label} | {fail.category} | {fail.reason} <br> *Rec: {fail.recommendation}* |\n")
            
            f.write("\n## 3. Small Object Bucket Counts\n")
            for bucket, count in stats["small_objects_buckets"].items():
                f.write(f"- **{bucket}**: {count} detections\n")
                
            f.write("\n## 4. Track Timeline Debugger\n")
            for tid, events in list(timeline_dict.items())[:10]: # Top 10 tracks
                f.write(f"### Track ID: {tid}\n")
                for ev in events:
                    f.write(f"- {ev}\n")
                f.write("\n")
                
        return report_path
