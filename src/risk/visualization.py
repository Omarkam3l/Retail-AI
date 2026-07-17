from typing import List, Dict, Tuple
import cv2
import numpy as np
from src.risk.types import RiskLevel
from src.common.types import TrackedPerson

def draw_risk_indicators(
    image: np.ndarray,
    risk_scores: Dict[int, float],
    risk_levels: Dict[int, RiskLevel],
    persons: List[TrackedPerson]
) -> None:
    """Annotates customer risk states and scores directly on the frame bounding boxes."""
    h_img, w_img = image.shape[:2]

    # Level colors: LOW (Green), MEDIUM (Yellow), HIGH (Red)
    colors = {
        RiskLevel.LOW: (0, 255, 0),
        RiskLevel.MEDIUM: (0, 165, 255),
        RiskLevel.HIGH: (0, 0, 255)
    }

    for person in persons:
        tid = person.track_id
        score = risk_scores.get(tid, 0.0)
        level = risk_levels.get(tid, RiskLevel.LOW)
        color = colors.get(level, (255, 255, 255))

        # Get coordinates
        x_min = int(person.bbox.x_min * w_img)
        y_max = int(person.bbox.y_max * h_img)

        # Draw risk tag text just below the bottom of the bounding box
        label = f"RISK: {level.value} ({score:.0f}%)"
        cv2.putText(
            image,
            label,
            (x_min, y_max + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )
