from typing import List, Dict, Tuple
import cv2
import numpy as np
from src.common.types import TrackedPerson, DetectedObject, AssociationState

def draw_associations(
    image: np.ndarray,
    association_map: Dict[int, Dict[int, AssociationState]],
    persons: List[TrackedPerson],
    objects: List[DetectedObject],
    line_color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1
) -> None:
    """Draws relationship linkage lines between associated customers and products/bags."""
    h_img, w_img = image.shape[:2]

    for p_id, object_states in association_map.items():
        person = next((p for p in persons if p.track_id == p_id), None)
        if not person:
            continue

        p_center = (int(person.bbox.center[0] * w_img), int(person.bbox.center[1] * h_img))

        for o_id, state in object_states.items():
            if state != AssociationState.ASSOCIATED:
                continue

            obj = next((o for o in objects if o.track_id == o_id), None)
            if not obj:
                continue

            o_center = (int(obj.bbox.center[0] * w_img), int(obj.bbox.center[1] * h_img))

            # Draw connection line
            cv2.line(image, p_center, o_center, line_color, thickness, cv2.LINE_AA)
            cv2.circle(image, o_center, thickness + 2, line_color, -1)

            # Draw relation text tag
            cv2.putText(
                image,
                f"Holding Obj {o_id}",
                (p_center[0] - 20, p_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                line_color,
                1,
                cv2.LINE_AA
            )
