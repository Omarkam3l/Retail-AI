import logging
from typing import List, Tuple, Optional, Any
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from src.tracking.interfaces import BaseTracker
from src.tracking.exceptions import ObjectTrackingError
from src.tracking.manager import TrackManager
from src.common.types import TrackedPerson, DetectedObject, BoundingBox, ClassLabel

logger = logging.getLogger("ByteTrackAdapter")

class TrackerArgs:
    """Namespace holder class matching Ultralytics' internal argument parser specifications."""
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: Optional[float] = None,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = False
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh if new_track_thresh is not None else track_high_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score


class ByteTrackDetections:
    """Custom detections wrapper that behaves like Ultralytics boxes matrix for slice operations."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xywh)

    def __getitem__(self, mask: np.ndarray) -> "ByteTrackDetections":
        return ByteTrackDetections(
            self.xywh[mask],
            self.conf[mask],
            self.cls[mask]
        )


class ByteTrackAdapter(BaseTracker):
    """Clean Architecture Adapter wrapping Ultralytics' ByteTrack engine."""

    # Map class labels to index integers
    LABEL_TO_IDX = {
        ClassLabel.PERSON: 0,
        ClassLabel.BACKPACK: 1,
        ClassLabel.HANDBAG: 2,
        ClassLabel.SHOPPING_BASKET: 3,
        ClassLabel.SHELF_ITEM: 4
    }

    # Map index integers back to ClassLabel enums
    IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

    def __init__(
        self,
        track_threshold: float = 0.5,
        match_threshold: float = 0.8,
        track_buffer: int = 30,
        track_manager: Optional[TrackManager] = None
    ) -> None:
        self._track_threshold = track_threshold
        self._match_threshold = match_threshold
        self._track_buffer = track_buffer
        self._manager = track_manager or TrackManager(max_occlusion_frames=track_buffer)
        
        self._tracker: Optional[BYTETracker] = None

    def initialize(self) -> None:
        try:
            logger.info("Initializing ByteTrack tracker adapter...")
            args = TrackerArgs(
                track_high_thresh=self._track_threshold,
                track_low_thresh=0.1,
                track_buffer=self._track_buffer,
                match_thresh=self._match_threshold
            )
            self._tracker = BYTETracker(args)
            self._manager.clear()
            logger.info("ByteTrack initialization completed successfully.")
        except Exception as e:
            raise ObjectTrackingError(f"Failed to initialize ByteTrack: {e}") from e

    def track(
        self, 
        frame: np.ndarray, 
        detections: List[DetectedObject]
    ) -> Tuple[List[TrackedPerson], List[DetectedObject]]:
        """Maps incoming detections, updates ByteTrack, and returns persistent Track items."""
        if self._tracker is None:
            raise ObjectTrackingError("Tracker is not initialized. Call initialize() first.")

        h_frame, w_frame = frame.shape[:2]

        # 1. Format inputs for ByteTrack update
        xywh_list = []
        conf_list = []
        cls_list = []

        for det in detections:
            # Bounding box converted to pixel coordinates
            x_min = det.bbox.x_min * w_frame
            y_min = det.bbox.y_min * h_frame
            x_max = det.bbox.x_max * w_frame
            y_max = det.bbox.y_max * h_frame
            
            # Convert to [x_center, y_center, width, height]
            w = x_max - x_min
            h = y_max - y_min
            xc = x_min + w / 2.0
            yc = y_min + h / 2.0

            xywh_list.append([xc, yc, w, h])
            conf_list.append(det.confidence)
            cls_list.append(self.LABEL_TO_IDX.get(det.class_label, 4))

        # Convert to numpy arrays
        xywh_arr = np.array(xywh_list, dtype=np.float32).reshape(-1, 4)
        conf_arr = np.array(conf_list, dtype=np.float32)
        cls_arr = np.array(cls_list, dtype=np.float32)

        det_obj = ByteTrackDetections(xywh_arr, conf_arr, cls_arr)

        try:
            # 2. Forward updates to ByteTrack
            tracker_output = self._tracker.update(det_obj)
        except Exception as e:
            raise ObjectTrackingError(f"Error during ByteTrack execution: {e}") from e

        # 3. Process tracker outputs & update TrackManager
        active_tracks = []
        for row in tracker_output:
            # Row elements: [x_min, y_min, x_max, y_max, track_id, score, cls, idx]
            x_min, y_min, x_max, y_max, track_id, score, cls_id, _ = row
            track_id = int(track_id)
            cls_id = int(cls_id)
            
            # Normalize bounding box back to [0.0, 1.0]
            normalized_bbox = BoundingBox(
                x_min=max(0.0, min(1.0, x_min / w_frame)),
                y_min=max(0.0, min(1.0, y_min / h_frame)),
                x_max=max(0.0, min(1.0, x_max / w_frame)),
                y_max=max(0.0, min(1.0, y_max / h_frame))
            )
            
            active_tracks.append((track_id, normalized_bbox, float(score)))

        self._manager.update(active_tracks)

        # 4. Construct domain lists based on updated states
        tracked_persons: List[TrackedPerson] = []
        tracked_objects: List[DetectedObject] = []

        for tid, meta in self._manager.get_active_tracks().items():
            # Retrieve original label index from detections if matching
            # For simplicity, we can assume ClassLabel based on index
            # Wait, let's look at how we mapped:
            # index 0 is PERSON.
            # other indices correspond to bags or shelf items.
            if tid in self._tracker.tracked_stracks:
                # Retrieve the original cls value from STrack
                strack = next((x for x in self._tracker.tracked_stracks if x.track_id == tid), None)
                cls_val = int(strack.cls) if strack is not None else 0
            else:
                cls_val = 0

            label = self.IDX_TO_LABEL.get(cls_val, ClassLabel.PERSON)

            if label == ClassLabel.PERSON:
                tracked_persons.append(
                    TrackedPerson(
                        track_id=tid,
                        bbox=meta.bbox,
                        confidence=meta.confidence,
                        velocity=meta.velocity,
                        age_frames=meta.age_frames
                    )
                )
            else:
                tracked_objects.append(
                    DetectedObject(
                        class_label=label,
                        bbox=meta.bbox,
                        confidence=meta.confidence,
                        track_id=tid
                    )
                )

        return tracked_persons, tracked_objects

    def shutdown(self) -> None:
        self._tracker = None
        self._manager.clear()
