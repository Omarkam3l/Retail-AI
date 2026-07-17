from typing import Tuple, List, Dict
import cv2
import numpy as np
from src.common.types import BoundingBox, Keypoint

def resize_letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resizes and pads an image using letterboxing to maintain aspect ratio.

    Args:
        image: Input NumPy BGR image.
        target_size: Tuple (width, height) for target dimensions.
        color: RGB padding color.

    Returns:
        A tuple containing:
        1. The letterboxed NumPy image.
        2. The scaling ratio applied.
        3. A tuple (pad_w, pad_h) representing the padding offset.
    """
    h_orig, w_orig = image.shape[:2]
    w_target, h_target = target_size

    # Scale ratio (new / old)
    r = min(w_target / w_orig, h_target / h_orig)

    # Compute unpadded size
    new_w, new_h = int(round(w_orig * r)), int(round(h_orig * r))
    dw, dh = w_target - new_w, h_target - new_h  # Total padding

    # Divide padding equally between top/bottom and left/right
    dw /= 2
    dh /= 2

    # Resize image
    if (w_orig, h_orig) != (new_w, new_h):
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = image

    # Pad image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, r, (left, top)


def draw_bounding_box(
    image: np.ndarray,
    bbox: BoundingBox,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Draws a 2D bounding box and text label directly onto the image."""
    h, w = image.shape[:2]
    # Denormalize coordinates if they are between [0, 1]
    x_min = int(bbox.x_min * w) if bbox.x_min <= 1.0 else int(bbox.x_min)
    y_min = int(bbox.y_min * h) if bbox.y_min <= 1.0 else int(bbox.y_min)
    x_max = int(bbox.x_max * w) if bbox.x_max <= 1.0 else int(bbox.x_max)
    y_max = int(bbox.y_max * h) if bbox.y_max <= 1.0 else int(bbox.y_max)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Draw label box
    tf = max(thickness - 1, 1)
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf)
    lbl_w, lbl_h = label_size[0]
    
    # Draw label background
    cv2.rectangle(image, (x_min, y_min - lbl_h - 4), (x_min + lbl_w, y_min), color, -1)
    cv2.putText(
        image, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), tf, cv2.LINE_AA
    )


def draw_pose_skeleton(
    image: np.ndarray,
    keypoints: Dict[int, Keypoint],
    skeleton_connections: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> None:
    """Draws keypoints and skeleton connection lines directly onto the image."""
    h, w = image.shape[:2]

    # Draw connection lines
    for p1, p2 in skeleton_connections:
        if p1 in keypoints and p2 in keypoints:
            k1 = keypoints[p1]
            k2 = keypoints[p2]
            
            if k1.confidence > 0.4 and k2.confidence > 0.4:
                pt1 = (int(k1.x * w), int(k1.y * h))
                pt2 = (int(k2.x * w), int(k2.y * h))
                cv2.line(image, pt1, pt2, color, thickness)

    # Draw joints
    for joint_id, k in keypoints.items():
        if k.confidence > 0.4:
            pt = (int(k.x * w), int(k.y * h))
            cv2.circle(image, pt, thickness + 1, (255, 255, 0), -1)
