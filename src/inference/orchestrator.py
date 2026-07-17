import logging
import time
from typing import Tuple, List, Optional, Dict
import numpy as np

from src.inference.interfaces import BaseInferencePipeline
from src.inference.exceptions import InferencePipelineError
from src.inference.context import ExecutionContext
from src.inference.event_bus import EventBus
from src.inference.profiler import PipelineProfiler
from src.detection.interfaces import BaseDetector
from src.tracking.interfaces import BaseTracker
from src.association.interfaces import BaseAssociationEngine
from src.behavior.interfaces import BaseBehaviorEngine
from src.common.types import FrameMetadata, DetectedObject, TrackedPerson
from src.alerts.types import Alert
from src.behavior.types import BehaviorFlag

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
        profiler: Optional[PipelineProfiler] = None
    ) -> None:
        self.camera_id = camera_id
        self._detector = detector
        self._tracker = tracker
        self._association_engine = association_engine
        self._behavior_engine = behavior_engine
        
        self.event_bus = event_bus or EventBus()
        self.profiler = profiler or PipelineProfiler()
        self._initialized = False

    def initialize(self) -> None:
        logger.info("Initializing PipelineOrchestrator components...")
        self._detector.initialize()
        self._tracker.initialize()
        # association and behavior engines are stateless or initialized via python init
        self._initialized = True
        logger.info("PipelineOrchestrator successfully initialized.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float
    ) -> Tuple[FrameMetadata, List[Alert]]:
        """Orchestrates detection, tracking, association, and behavior analysis sequentially."""
        if not self._initialized:
            raise InferencePipelineError("PipelineOrchestrator is not initialized.")

        # Create localized ExecutionContext
        ctx = ExecutionContext(self.camera_id, frame_index, timestamp_ms)
        
        # 1. Detection Stage
        detections: List[DetectedObject] = []
        try:
            start = time.perf_counter()
            detections = self._detector.detect(frame)
            dur = (time.perf_counter() - start) * 1000.0
            ctx.record_stage_latency("detection", dur)
            self.profiler.record_latency("detection", dur)
        except Exception as e:
            logger.error(f"Detection stage failure: {e}", exc_info=True)

        # 2. Tracking Stage
        tracked_persons: List[TrackedPerson] = []
        tracked_objects: List[DetectedObject] = []
        try:
            start = time.perf_counter()
            tracked_persons, tracked_objects = self._tracker.track(frame, detections)
            dur = (time.perf_counter() - start) * 1000.0
            ctx.record_stage_latency("tracking", dur)
            self.profiler.record_latency("tracking", dur)
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
            start = time.perf_counter()
            if hasattr(self._association_engine, "mock_timestamp_ms"):
                self._association_engine.mock_timestamp_ms = timestamp_ms
            associations = self._association_engine.associate(frame, tracked_persons, tracked_objects)
            dur = (time.perf_counter() - start) * 1000.0
            ctx.record_stage_latency("association", dur)
            self.profiler.record_latency("association", dur)
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
            start = time.perf_counter()
            behavior_flags = self._behavior_engine.analyze(metadata, assoc_events)
            dur = (time.perf_counter() - start) * 1000.0
            ctx.record_stage_latency("behavior", dur)
            self.profiler.record_latency("behavior", dur)
        except Exception as e:
            logger.error(f"Behavior analysis stage failure: {e}", exc_info=True)

        # Publish behavior flags to EventBus
        for flag in behavior_flags:
            self.event_bus.publish("behavior_flag", flag)

        # Stage 5: Alerting & Risk (Placeholder for Sprint 6)
        alerts: List[Alert] = []

        return metadata, alerts

    def shutdown(self) -> None:
        logger.info("Shutting down PipelineOrchestrator...")
        self._detector.shutdown()
        self._tracker.shutdown()
        if hasattr(self._association_engine, "shutdown"):
            self._association_engine.shutdown()
        if hasattr(self._behavior_engine, "shutdown"):
            self._behavior_engine.shutdown()
        self.profiler.reset()
        self._initialized = False
        logger.info("PipelineOrchestrator shutdown complete.")
