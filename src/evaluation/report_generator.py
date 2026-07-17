import os
import logging
from typing import Dict, Any, List, Optional
from src.evaluation.types import OverallEvaluationResult

logger = logging.getLogger("ReportGenerator")

class ReportGenerator:
    """Generates evaluation reports in Markdown (and stubs for HTML/PDF)."""

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_markdown(
        self,
        result: OverallEvaluationResult,
        title: str = "Evaluation Report",
        confusion_matrix_md: str = "",
        benchmark_table: List[Dict[str, Any]] = None,
        recommendations: List[str] = None
    ) -> str:
        """Generates a Markdown report string and writes it to disk."""
        lines = [f"# {title}", ""]

        lines.append("## Summary")
        lines.append(f"- Execution Time: {result.execution_time_seconds:.2f}s")
        lines.append("")

        # Detection Metrics
        if result.detection:
            d = result.detection
            lines.append("## Detection Metrics")
            lines.append(f"| Metric | Value |")
            lines.append(f"| --- | --- |")
            lines.append(f"| Precision | {d.precision:.4f} |")
            lines.append(f"| Recall | {d.recall:.4f} |")
            lines.append(f"| F1 | {d.f1:.4f} |")
            lines.append(f"| mAP@50 | {d.mAP50:.4f} |")
            lines.append(f"| mAP@50-95 | {d.mAP50_95:.4f} |")
            lines.append("")

        # Tracking Metrics
        if result.tracking:
            t = result.tracking
            lines.append("## Tracking Metrics")
            lines.append(f"| Metric | Value |")
            lines.append(f"| --- | --- |")
            lines.append(f"| MOTA | {t.mota:.4f} |")
            lines.append(f"| MOTP | {t.motp:.4f} |")
            lines.append(f"| IDF1 | {t.idf1:.4f} |")
            lines.append(f"| ID Switches | {t.id_switches} |")
            lines.append(f"| Fragmentations | {t.fragmentations} |")
            lines.append("")

        # Behavior Metrics
        if result.behavior and result.behavior.per_behavior:
            lines.append("## Behavior Metrics")
            lines.append("| Behavior | Precision | Recall | F1 |")
            lines.append("| --- | --- | --- | --- |")
            for beh, vals in result.behavior.per_behavior.items():
                lines.append(f"| {beh} | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} |")
            lines.append("")

        # Risk Metrics
        if result.risk:
            r = result.risk
            lines.append("## Risk Metrics")
            lines.append(f"| Metric | Value |")
            lines.append(f"| --- | --- |")
            lines.append(f"| Precision | {r.precision:.4f} |")
            lines.append(f"| Recall | {r.recall:.4f} |")
            lines.append(f"| Avg Delay (ms) | {r.average_delay_ms:.2f} |")
            lines.append(f"| Escalation Accuracy | {r.escalation_accuracy:.4f} |")
            lines.append("")

        # Alert Metrics
        if result.alerts:
            a = result.alerts
            lines.append("## Alert Metrics")
            lines.append(f"| Metric | Value |")
            lines.append(f"| --- | --- |")
            lines.append(f"| Precision | {a.precision:.4f} |")
            lines.append(f"| Recall | {a.recall:.4f} |")
            lines.append(f"| Duplicate Alerts | {a.duplicate_alerts} |")
            lines.append(f"| Missed Alerts | {a.missed_alerts} |")
            lines.append("")

        # Confusion Matrix
        if confusion_matrix_md:
            lines.append("## Confusion Matrix")
            lines.append(confusion_matrix_md)
            lines.append("")

        # Benchmark Comparison
        if benchmark_table:
            lines.append("## Benchmark Comparison")
            lines.append("| Name | Model | Config | Det F1 | Trk MOTA | Time |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in benchmark_table:
                lines.append(f"| {row.get('name','')} | {row.get('model','')} | {row.get('config','')} | {row.get('det_f1','')} | {row.get('trk_mota','')} | {row.get('time_s','')} |")
            lines.append("")

        # Recommendations
        if recommendations:
            lines.append("## Recommendations")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        report_text = "\n".join(lines)
        report_path = os.path.join(self._output_dir, f"{title.replace(' ', '_').lower()}.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Markdown report generated at {report_path}.")
        return report_text

    def generate_html(self, markdown_content: str) -> str:
        """Stub for HTML report generation. Returns basic HTML wrapper."""
        html = f"<html><body><pre>{markdown_content}</pre></body></html>"
        return html

    def generate_pdf(self, markdown_content: str) -> str:
        """Stub for PDF report generation. Returns a placeholder path."""
        logger.info("PDF generation requires weasyprint. Stub returning placeholder.")
        return "report.pdf (stub)"
