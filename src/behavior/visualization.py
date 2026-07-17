from typing import List, Tuple
import cv2
import numpy as np
from src.behavior.types import BehaviorFlag
from src.common.types import TrackedPerson

def draw_behavior_alerts(
    image: np.ndarray,
    alerts: List[BehaviorFlag],
    persons: List[TrackedPerson],
    alert_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3
) -> None:
    """Highlights flagged customer bounding boxes in red and displays behavior alerts."""
    h_img, w_img = image.shape[:2]

    # Map track_id -> list of active alerts
    track_alerts = {}
    for alert in alerts:
        if alert.track_id not in track_alerts:
            track_alerts[alert.track_id] = []
        track_alerts[alert.track_id].append(alert.behavior_type)

    for person in persons:
        if person.track_id not in track_alerts:
            continue

        # Draw red alert bounding box
        x_min = int(person.bbox.x_min * w_img)
        y_min = int(person.bbox.y_min * h_img)
        x_max = int(person.bbox.x_max * w_img)
        y_max = int(person.bbox.y_max * h_img)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), alert_color, thickness)

        # Draw warning banners
        alert_str = " | ".join(track_alerts[person.track_id])
        label = f"WARNING: {alert_str}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lbl_w, lbl_h = label_size[0]
        
        cv2.rectangle(image, (x_min, y_min - lbl_h - 6), (x_min + lbl_w + 4, y_min), alert_color, -1)
        cv2.putText(
            image,
            label,
            (x_min + 2, y_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
