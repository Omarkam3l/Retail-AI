import logging
import prometheus_client
from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger("Observability")

def get_counter(name: str, documentation: str, labelnames: list[str]) -> prometheus_client.Counter:
    """Helper to safely define or retrieve a Counter to prevent duplicate registration errors."""
    reg_names = prometheus_client.REGISTRY._names_to_collectors
    if name in reg_names:
        return reg_names[name]
    # Suffix check
    total_name = f"{name}_total"
    if total_name in reg_names:
        return reg_names[total_name]
    return prometheus_client.Counter(name, documentation, labelnames)


def get_histogram(name: str, documentation: str, labelnames: list[str]) -> prometheus_client.Histogram:
    """Helper to safely define or retrieve a Histogram to prevent duplicate registration errors."""
    reg_names = prometheus_client.REGISTRY._names_to_collectors
    if name in reg_names:
        return reg_names[name]
    return prometheus_client.Histogram(name, documentation, labelnames)


# Initialize Prometheus Metrics (Thread-safe, non-blocking)
PIPELINE_FRAMES = get_counter(
    "pipeline_processed_frames_total", 
    "Total number of processed camera frames", 
    ["camera_id"]
)

STAGE_LATENCY = get_histogram(
    "pipeline_stage_latency_milliseconds", 
    "Histogram of processing durations per cascade stage in ms", 
    ["camera_id", "stage_name"]
)

ALERT_COUNT = get_counter(
    "pipeline_alerts_total", 
    "Total security alerts raised", 
    ["camera_id", "alert_level"]
)

# Initialize OpenTelemetry Tracer
trace.set_tracer_provider(TracerProvider())
tracer: Tracer = trace.get_tracer("retail-ai-pipeline")

def start_metrics_server(port: int = 8000) -> None:
    """Starts a non-blocking Prometheus metrics exporter server on a background port."""
    try:
        prometheus_client.start_http_server(port)
        logger.info(f"Prometheus HTTP metrics exporter started on port {port}.")
    except Exception as e:
        logger.warning(f"Could not start metrics exporter server (perhaps port is occupied): {e}")
