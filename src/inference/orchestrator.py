import logging
import time
from typing import Tuple, List, Optional, Dict
import numpy as np

from src.inference.interfaces import BaseInferencePipeline
from src.inference.exceptions import InferencePipelineError
from src.inference.context import ExecutionContext
from src.inference.event_bus import EventBus
from src.inference.profiler import PipelineProfiler
from src.common.observability import PIPELINE_FRAMES, STAGE_LATENCY, ALERT_COUNT, tracer
from src.detection.interfaces import BaseDetector
from src.tracking.interfaces import BaseTracker
from src.association.interfaces import BaseAssociationEngine
from src.behavior.interfaces import BaseBehaviorEngine
from src.common.types import FrameMetadata, DetectedObject, TrackedPerson
from src.alerts.types import Alert
from src.behavior.types import BehaviorFlag

# Optional future integrations
try:
    from src.risk.interfaces import BaseRiskEngine
except ImportError:
    BaseRiskEngine = None

try:
    from src.alerts.interfaces import BaseAlertEngine
except ImportError:
    BaseAlertEngine = None

logger = logging.getLogger("PipelineOrchestrator")

class PipelineOrchestrator(BaseInferencePipeline):
    """Production-ready coordinator organizing the computer vision cascade stages."""

    def __init__(
        self,
        camera_id: str,
        detector: BaseDetector,
        tracker: BaseTracker,
        association_engine: BaseAssociationEngine,
        behavior_engine: BaseBehaviorEngine,
        event_bus: Optional[EventBus] = None,
        profiler: Optional[PipelineProfiler] = None,
        risk_engine: Optional[BaseRiskEngine] = None,
        alert_engine: Optional[BaseAlertEngine] = None
    ) -> None:
        self.camera_id = camera_id
        self._detector = detector
        self._tracker = tracker
        self._association_engine = association_engine
        self._behavior_engine = behavior_engine
        
        self.event_bus = event_bus or EventBus()
        self.profiler = profiler or PipelineProfiler()
        self._risk_engine = risk_engine
        self._alert_engine = alert_engine
        self._initialized = False

    def initialize(self) -> None:
        logger.info("Initializing PipelineOrchestrator components...")
        self._detector.initialize()
        self._tracker.initialize()
        if self._risk_engine is not None and hasattr(self._risk_engine, "initialize"):
            self._risk_engine.initialize()
        if self._alert_engine is not None and hasattr(self._alert_engine, "initialize"):
            self._alert_engine.initialize()
        self._initialized = True
        logger.info("PipelineOrchestrator successfully initialized.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float
    ) -> Tuple[FrameMetadata, List[Alert]]:
        """Orchestrates detection, tracking, association, behavior, risk, and alerts sequentially."""
        if not self._initialized:
            raise InferencePipelineError("PipelineOrchestrator is not initialized.")

        # Increment Prometheus global frame metric
        PIPELINE_FRAMES.labels(camera_id=self.camera_id).inc()

        # Start OpenTelemetry root processing span
        with tracer.start_as_current_span("process_frame") as span:
            span.set_attribute("camera_id", self.camera_id)
            span.set_attribute("frame_index", frame_index)

            # Create localized ExecutionContext
            ctx = ExecutionContext(self.camera_id, frame_index, timestamp_ms)
            
            # 1. Detection Stage
            detections: List[DetectedObject] = []
            try:
                with tracer.start_as_current_span("detection_stage"):
                    start = time.perf_counter()
                    detections = self._detector.detect(frame)
                    dur = (time.perf_counter() - start) * 1000.0
                    ctx.record_stage_latency("detection", dur)
                    self.profiler.record_latency("detection", dur)
                    STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="detection").observe(dur)
            except Exception as e:
                logger.error(f"Detection stage failure: {e}", exc_info=True)

            # 2. Tracking Stage
            tracked_persons: List[TrackedPerson] = []
            tracked_objects: List[DetectedObject] = []
            try:
                with tracer.start_as_current_span("tracking_stage"):
                    start = time.perf_counter()
                    tracked_persons, tracked_objects = self._tracker.track(frame, detections)
                    dur = (time.perf_counter() - start) * 1000.0
                    ctx.record_stage_latency("tracking", dur)
                    self.profiler.record_latency("tracking", dur)
                    STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="tracking").observe(dur)
            except Exception as e:
                logger.error(f"Tracking stage failure: {e}", exc_info=True)

            # Build FrameMetadata object
            metadata = FrameMetadata(
                camera_id=self.camera_id,
                timestamp_ms=timestamp_ms,
                frame_index=frame_index,
                persons=tracked_persons,
                objects=tracked_objects
            )

            # 3. Association Stage
            associations = {}
            try:
                with tracer.start_as_current_span("association_stage"):
                    start = time.perf_counter()
                    if hasattr(self._association_engine, "mock_timestamp_ms"):
                        self._association_engine.mock_timestamp_ms = timestamp_ms
                    associations = self._association_engine.associate(frame, tracked_persons, tracked_objects)
                    dur = (time.perf_counter() - start) * 1000.0
                    ctx.record_stage_latency("association", dur)
                    self.profiler.record_latency("association", dur)
                    STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="association").observe(dur)
            except Exception as e:
                logger.error(f"Association stage failure: {e}", exc_info=True)

            # Extract association events from engine queue
            assoc_events = []
            if hasattr(self._association_engine, "get_events"):
                assoc_events = self._association_engine.get_events()

            # Publish association events to EventBus
            for event in assoc_events:
                self.event_bus.publish("association_event", event)

            # 4. Behavior Analysis Stage
            behavior_flags: List[BehaviorFlag] = []
            try:
                with tracer.start_as_current_span("behavior_stage"):
                    start = time.perf_counter()
                    behavior_flags = self._behavior_engine.analyze(metadata, assoc_events)
                    dur = (time.perf_counter() - start) * 1000.0
                    ctx.record_stage_latency("behavior", dur)
                    self.profiler.record_latency("behavior", dur)
                    STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="behavior").observe(dur)
            except Exception as e:
                logger.error(f"Behavior analysis stage failure: {e}", exc_info=True)

            # Publish behavior flags to EventBus
            for flag in behavior_flags:
                self.event_bus.publish("behavior_flag", flag)

            # 5. Risk Assessment Stage (Optional Integration)
            risk_scores: Dict[int, float] = {}
            risk_events = []
            if self._risk_engine is not None:
                try:
                    with tracer.start_as_current_span("risk_stage"):
                        start = time.perf_counter()
                        # Run risk calculations for each person
                        for person in tracked_persons:
                            # Filter flags for this track ID
                            person_flags = [f for f in behavior_flags if f.track_id == person.track_id]
                            self._risk_engine.calculate_risk(person.track_id, person_flags)
                        
                        risk_scores = self._risk_engine.get_all_risk_scores()
                        if hasattr(self._risk_engine, "get_events"):
                            risk_events = self._risk_engine.get_events()
                            
                        dur = (time.perf_counter() - start) * 1000.0
                        ctx.record_stage_latency("risk", dur)
                        self.profiler.record_latency("risk", dur)
                        STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="risk").observe(dur)
                except Exception as e:
                    logger.error(f"Risk stage failure: {e}", exc_info=True)

            # Publish risk events to EventBus
            for rev in risk_events:
                self.event_bus.publish("risk_event", rev)

            # 6. Alert & Evidence Stage (Optional Integration)
            alerts: List[Alert] = []
            if self._alert_engine is not None:
                try:
                    with tracer.start_as_current_span("alerts_stage"):
                        start = time.perf_counter()
                        
                        # Ingest risk events if supported
                        if hasattr(self._alert_engine, "ingest_risk_events") and risk_events:
                            self._alert_engine.ingest_risk_events(risk_events)
                            
                        alerts = self._alert_engine.evaluate(risk_scores, metadata)
                        
                        dur = (time.perf_counter() - start) * 1000.0
                        ctx.record_stage_latency("alerts", dur)
                        self.profiler.record_latency("alerts", dur)
                        STAGE_LATENCY.labels(camera_id=self.camera_id, stage_name="alerts").observe(dur)
                        
                        # Increment Prometheus alert metric
                        for alert in alerts:
                            ALERT_COUNT.labels(camera_id=self.camera_id, alert_level=alert.level.value).inc()
                            self.event_bus.publish("alert_event", alert)
                except Exception as e:
                    logger.error(f"Alert stage failure: {e}", exc_info=True)

            return metadata, alerts

    def shutdown(self) -> None:
        logger.info("Shutting down PipelineOrchestrator...")
        self._detector.shutdown()
        self._tracker.shutdown()
        if hasattr(self._association_engine, "shutdown"):
            self._association_engine.shutdown()
        if hasattr(self._behavior_engine, "shutdown"):
            self._behavior_engine.shutdown()
        if self._risk_engine is not None and hasattr(self._risk_engine, "shutdown"):
            self._risk_engine.shutdown()
        if self._alert_engine is not None and hasattr(self._alert_engine, "shutdown"):
            self._alert_engine.shutdown()
        self.profiler.reset()
        self._initialized = False
        logger.info("PipelineOrchestrator shutdown complete.")
