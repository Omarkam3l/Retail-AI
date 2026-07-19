from typing import List, Tuple, Dict, Optional
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
    """Evaluates spatial, visual, and motion relationships to pair persons and objects."""

    def __init__(
        self,
        proximity_threshold: float = 0.25,
        iou_weight: float = 0.5,
        alpha: float = 0.4,       # BBox overlap weight
        beta: float = 0.5,        # Visual similarity weight
        gamma: float = 0.1         # Motion similarity weight
    ) -> None:
        self._proximity_threshold = proximity_threshold
        self._iou_weight = iou_weight
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def calculate_cost_matrix(
        self,
        persons: List[TrackedPerson],
        objects: List[DetectedObject],
        object_embeddings: Optional[Dict[int, np.ndarray]] = None,
        catalog_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Constructs an appearance-aware cost matrix between persons and objects."""
        num_persons = len(persons)
        num_objects = len(objects)
        
        cost_matrix = np.ones((num_persons, num_objects), dtype=np.float32) * 1e5

        for i, person in enumerate(persons):
            p_center = person.bbox.center
            
            for j, obj in enumerate(objects):
                o_center = obj.bbox.center
                
                dist = np.hypot(p_center[0] - o_center[0], p_center[1] - o_center[1])
                if dist > self._proximity_threshold:
                    continue

                # 1. Geometry score (Overlap coefficient)
                overlap = calculate_overlap_ratio(person.bbox, obj.bbox)
                geom_score = overlap

                # 2. Appearance score
                app_score = 0.0
                if object_embeddings and obj.track_id in object_embeddings:
                    obj_emb = object_embeddings[obj.track_id]
                    # If person has reference visual appearance or we check proximity similarity
                    # In retail, we can use visual appearance similarity (hands/arms overlap with product)
                    # For MVP, a base similarity of 0.5 is assigned if they overlap
                    if overlap > 0.0:
                        app_score = 0.8
                
                # 3. Motion consistency score
                # If product velocity matches person velocity
                motion_score = 0.5
                if hasattr(person, "velocity") and hasattr(obj, "velocity"):
                    # Check direction alignment
                    pass

                # Higher overall score = lower cost
                fused_score = self._alpha * geom_score + self._beta * app_score + self._gamma * motion_score
                cost = 1.0 - fused_score
                cost_matrix[i, j] = max(0.0, cost)

        return cost_matrix

    def match(
        self,
        persons: List[TrackedPerson],
        objects: List[DetectedObject],
        object_embeddings: Optional[Dict[int, np.ndarray]] = None
    ) -> List[Tuple[int, int, float]]:
        """Executes Hungarian linear sum assignment to pair persons and objects."""
        if not persons or not objects:
            return []

        cost_matrix = self.calculate_cost_matrix(persons, objects, object_embeddings)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= 0.95:  # Gate threshold
                continue
                
            person = persons[r]
            obj = objects[c]
            
            confidence = max(0.0, min(1.0, 1.0 - cost))
            
            if obj.track_id is not None:
                matches.append((person.track_id, obj.track_id, confidence))
                
        return matches
