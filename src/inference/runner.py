import logging
import time
from typing import List, Tuple, Callable, Optional
from src.inference.interfaces import BaseInferencePipeline
from src.ingestion.replay import VideoReplayEngine
from src.common.types import FrameMetadata
from src.alerts.types import Alert

logger = logging.getLogger("ReplayRunner")

class ReplayRunner:
    """Synchronous pipeline executor driving frame processing from pre-recorded source streams."""

    def __init__(
        self,
        replay_engine: VideoReplayEngine,
        pipeline: BaseInferencePipeline
    ) -> None:
        self._replay = replay_engine
        self._pipeline = pipeline

    def run(
        self,
        on_frame_processed: Optional[Callable[[FrameMetadata, List[Alert]], None]] = None
    ) -> List[Tuple[FrameMetadata, List[Alert]]]:
        """Runs the entire video file frame-by-frame synchronously.

        Returns:
            A list of (FrameMetadata, list of Alerts) generated during the run.
        """
        logger.info("Starting synchronous offline video replay pipeline execution...")
        self._replay.initialize()
        self._pipeline.initialize()
        
        results: List[Tuple[FrameMetadata, List[Alert]]] = []

        try:
            while True:
                success, frame, metadata = self._replay.next_frame()
                if not success or frame is None:
                    # End of file reached
                    break
                    
                frame_idx = metadata["frame_index"]
                timestamp_ms = metadata["timestamp_ms"]
                
                # Execute pipeline synchronously
                frame_meta, alerts = self._pipeline.process_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    timestamp_ms=timestamp_ms
                )
                
                results.append((frame_meta, alerts))
                
                if on_frame_processed is not None:
                    on_frame_processed(frame_meta, alerts)
                    
            logger.info("Synchronous replay pipeline execution completed.")
            
        finally:
            self._replay.release()
            self._pipeline.shutdown()

        return results
