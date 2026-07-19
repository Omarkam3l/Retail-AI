from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.types import BoundingBox, TrackedPerson, DetectedObject

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates Intersection over Union between two bounding boxes."""
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    union_area = float(box1_area + box2_area - intersection_area)

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def calculate_overlap_ratio(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates Overlap Coefficient (Intersection over Minimum Area) between two bounding boxes."""
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    
    min_area = min(box1_area, box2_area)

    if min_area <= 0.0:
        return 0.0
    return intersection_area / min_area


class SpatialMatcher:
    """Evaluates spatial relationships and resolves overlap conflicts using Hungarian matching."""

    def __init__(self, proximity_threshold: float = 0.25, iou_weight: float = 0.6) -> None:
        self._proximity_threshold = proximity_threshold
        self._iou_weight = iou_weight

    def calculate_cost_matrix(
        self,
        persons: List[TrackedPerson],
        objects: List[DetectedObject]
    ) -> np.ndarray:
        """Constructs a cost matrix between person hand region/centroid and objects."""
        num_persons = len(persons)
        num_objects = len(objects)
        
        # Matrix shape: (num_persons, num_objects)
        cost_matrix = np.ones((num_persons, num_objects), dtype=np.float32) * 1e5

        for i, person in enumerate(persons):
            p_center = person.bbox.center
            
            for j, obj in enumerate(objects):
                o_center = obj.bbox.center
                
                # Compute Euclidean distance
                dist = np.hypot(p_center[0] - o_center[0], p_center[1] - o_center[1])
                
                if dist > self._proximity_threshold:
                    # Too far away, keep cost high (gated)
                    continue

                # Compute overlap ratio (overlap coefficient)
                iou = calculate_overlap_ratio(person.bbox, obj.bbox)
                
                # Cost is lower for closer proximity and higher overlap
                cost = (1.0 - self._iou_weight) * dist + self._iou_weight * (1.0 - iou)
                cost_matrix[i, j] = cost

        return cost_matrix

    def match(
        self,
        persons: List[TrackedPerson],
        objects: List[DetectedObject]
    ) -> List[Tuple[int, int, float]]:
        """Executes Hungarian linear sum assignment to pair persons and objects.

        Returns:
            A list of tuples: (person_track_id, object_track_id, match_confidence)
        """
        if not persons or not objects:
            return []

        cost_matrix = self.calculate_cost_matrix(persons, objects)
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= 1e4:
                # Discard gated matches
                continue
                
            person = persons[r]
            obj = objects[c]
            
            # Confidence is inversely proportional to cost
            confidence = max(0.0, min(1.0, 1.0 - cost))
            
            # Since objects must be tracked, assert object has track_id
            if obj.track_id is not None:
                matches.append((person.track_id, obj.track_id, confidence))
                
        return matches
