from typing import List, Tuple
import cv2
import numpy as np
from src.common.types import TrackedPerson
from src.tracking.manager import TrackManager

def draw_track_trails(
    image: np.ndarray,
    tracked_persons: List[TrackedPerson],
    manager: TrackManager,
    trail_color: Tuple[int, int, int] = (255, 0, 255),
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Annotates persistent tracking bounding boxes and trajectory trails on the frame."""
    h_img, w_img = image.shape[:2]

    for person in tracked_persons:
        # Bbox coords
        x_min = int(person.bbox.x_min * w_img)
        y_min = int(person.bbox.y_min * h_img)
        x_max = int(person.bbox.x_max * w_img)
        y_max = int(person.bbox.y_max * h_img)

        # Draw bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), bbox_color, thickness)

        # Draw tracking label (ID + confidence)
        label = f"ID: {person.track_id} ({person.confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lbl_w, lbl_h = label_size[0]
        cv2.rectangle(image, (x_min, y_min - lbl_h - 4), (x_min + lbl_w, y_min), bbox_color, -1)
        cv2.putText(
            image, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw trajectory trails from history
        meta = manager.get_track_metadata(person.track_id)
        if meta and len(meta.history) >= 2:
            centers = []
            for bbox in meta.history:
                cx = int(bbox.center[0] * w_img)
                cy = int(bbox.center[1] * h_img)
                centers.append((cx, cy))
                
            # Draw line segments
            for i in range(len(centers) - 1):
                cv2.line(image, centers[i], centers[i + 1], trail_color, thickness)
                cv2.circle(image, centers[i + 1], thickness, trail_color, -1)
