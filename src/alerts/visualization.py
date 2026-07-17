from typing import List, Tuple
import cv2
import numpy as np
from src.alerts.types import Alert
from src.common.types import TrackedPerson, AlertLevel

def draw_alerts_overlay(
    image: np.ndarray,
    active_alerts: List[Alert],
    persons: List[TrackedPerson],
    flashing_color: Tuple[int, int, int] = (0, 0, 255)
) -> None:
    """Draws security alert tags and flashing overlays on camera feeds."""
    h_img, w_img = image.shape[:2]

    # Map track_id -> AlertLevel
    track_levels = {}
    for alert in active_alerts:
        track_levels[alert.track_id] = alert.level

    for person in persons:
        tid = person.track_id
        if tid not in track_levels:
            continue

        level = track_levels[tid]
        
        # Get coordinates
        x_min = int(person.bbox.x_min * w_img)
        y_min = int(person.bbox.y_min * h_img)
        x_max = int(person.bbox.x_max * w_img)
        y_max = int(person.bbox.y_max * h_img)

        # Draw thick flashing red alert box around shopper
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), flashing_color, 4)

        # Draw ALERT banner at the top of the box
        label = f"INCIDENT: {level.value} RISK"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        lbl_w, lbl_h = label_size[0]
        
        cv2.rectangle(image, (x_min, y_min - lbl_h - 8), (x_min + lbl_w + 6, y_min), flashing_color, -1)
        cv2.putText(
            image,
            label,
            (x_min + 3, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
