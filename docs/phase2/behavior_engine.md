# Behavior Rules Engine - Phase 2 Design Review

This document defines the mathematical models, spatial logic, and heuristics used by the Behavior Analysis Engine to detect concealment gestures and suspicious retail behaviors.

---

## 1. Mathematical Formulations & Spatial Relations

The Behavior Engine acts on 2D image coordinates, normalized between $[0, 1]$ relative to frame dimensions. Let $I(t)$ be the video frame at time $t$.

### 1.1 Keypoint Definitions
Let a tracked person $P_i$ at frame $t$ be defined by their tracking ID $i$ and their estimated 2D pose keypoints:
$$K_i(t) = \{k_{i, j}(t) \mid j \in [1, 17]\}$$
Where:
* $k_{i, 9}$: Left Wrist coordinate $(x_{lw}, y_{lw})$
* $k_{i, 10}$: Right Wrist coordinate $(x_{rw}, y_{rw})$
* $k_{i, 11}$: Left Hip coordinate $(x_{lh}, y_{lh})$
* $k_{i, 12}$: Right Hip coordinate $(x_{rh}, y_{rh})$

### 1.2 Bounding Box Definitions
Let an object detection bounding box $B$ be defined by its coordinates:
$$B = [x_{min}, y_{min}, x_{max}, y_{max}]$$
We track:
* $B_{item}(t)$: Bounding box of a product item.
* $B_{bag}(t)$: Bounding box of a detected backpack, handbag, or purse.
* $B_{person}(t)$: Bounding box of the person.

---

## 2. Suspicious Behavior Rules

```text
               [Aisle Interaction]
                        │
                        ▼
           (Hand Intersects Shelf Item)
                        │
                        ▼
         ┌──────────────┴──────────────┐
         ▼                             ▼
 [Wrist Close to Hip]        [Wrist Intersects Bag]
         │                             │
         ▼                             ▼
 (Item Disappears in Pocket)   (Item Disappears in Bag)
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
            [Alert State Triggered]
```

### Rule 1: Pocket Concealment Heuristic
Pocket concealment occurs when a customer grabs an item and moves their hand to their pocket or waist region, where the item disappears.

1. **Item Grab Detection**:
   The distance between a wrist keypoint $k_{i, wrist}(t)$ (where $wrist \in \{9, 10\}$) and a detected item $B_{item}(t)$ falls below a threshold:
   $$\text{dist}(k_{i, wrist}(t), \text{center}(B_{item}(t))) < \theta_{grab}$$
   This sets the state of the item to **"Grabbed by ID $i$"**.

2. **Wrist-to-Hip Proximity**:
   The distance between the grasping wrist keypoint and the corresponding hip keypoint $k_{i, hip}(t)$ (where $hip \in \{11, 12\}$) falls below a threshold:
   $$\text{dist}(k_{i, wrist}(t), k_{i, hip}(t)) < \theta_{pocket}$$
   Where $\theta_{pocket}$ is dynamically scaled based on the person's height:
   $$\theta_{pocket} = 0.15 \times \text{height}(B_{person}(t))$$

3. **Item Disappearance**:
   The object detector loses the track of $B_{item}(t)$ (confidence scores drop below $0.20$), while the wrist is in proximity to the hip:
   $$\text{confidence}(B_{item}(t)) \to 0 \quad \text{and} \quad \text{dist}(k_{i, wrist}(t), k_{i, hip}(t)) < \theta_{pocket}$$
   If the item does not reappear within $N_{confirm} = 30$ frames (2 seconds at 15 FPS), the state machine triggers a **Pocket Concealment Alert**.

---

### Rule 2: Bag/Backpack Concealment Heuristic
Bag concealment occurs when an item is placed directly inside a personal carrying container.

1. **Bag Association**:
   A bag $B_{bag}(t)$ is classified as a personal container belonging to person $P_i$ if:
   - The Intersection over Union (IoU) of $B_{bag}(t)$ and $B_{person}(t)$ is consistently high:
     $$\text{IoU}(B_{bag}(t), B_{person}(t)) > 0.40 \quad \text{for} \quad \Delta t > 10\text{ seconds}$$
   - The bag does not possess specific color/visual markers of store-provided shopping baskets.

2. **Grasp & Insertion**:
   - The customer grabs an item (Rule 1.1).
   - The wrist keypoint $k_{i, wrist}(t)$ intersects with the bag boundary $B_{bag}(t)$:
     $$k_{i, wrist}(t) \in B_{bag}(t)$$
   - The item bounding box $B_{item}(t)$ overlaps with the bag:
     $$\text{Intersection}(B_{item}(t), B_{bag}(t)) / \text{Area}(B_{item}(t)) > 0.70$$
   - The item detection is lost inside the bag boundary and fails to reappear for 45 frames (3 seconds). This triggers a **Bag Concealment Alert**.

---

### Rule 3: Loitering in High-Shrink Zones
1. **Zone Definition**:
   A polygon $Z$ is defined on the frame coordinate space by the store manager:
   $$Z = \{(x_1, y_1), (x_2, y_2), \ldots, (x_m, y_m)\}$$

2. **Presence Verification**:
   A person $P_i$ has their bottom-center bounding box coordinate $C_i(t) = \left( \frac{x_{min}+x_{max}}{2}, y_{max} \right)$ inside the polygon:
   $$C_i(t) \in Z$$

3. **Duration Threshold**:
   The person remains in the zone without significant translation movement:
   $$\Delta T_{presence} = t_{current} - t_{entry} > \theta_{loiter} \quad \text{and} \quad \text{Velocity}(C_i) < \theta_{velocity}$$
   Where $\theta_{loiter} = 180\text{ seconds}$ and velocity is measured as moving average distance over 300 frames. This triggers a **Loitering Notification**.

---

## 3. Pseudocode: Tracking Spatial Relations Over Time

```python
class BehaviorEngine:
    def __init__(self, fps=15):
        self.fps = fps
        self.grab_threshold = 0.08  # Normalized distance
        self.confirm_frames = 2 * fps  # 2 seconds
        
    def process_frame_data(self, tracked_persons, detected_items, active_tracks):
        alerts = []
        for person in tracked_persons:
            pid = person.track_id
            wrists = [person.keypoints.left_wrist, person.keypoints.right_wrist]
            hips = [person.keypoints.left_hip, person.keypoints.right_hip]
            height = person.bbox.y_max - person.bbox.y_min
            pocket_threshold = 0.15 * height
            
            # 1. Update hand inventory state
            for wrist_idx, wrist in enumerate(wrists):
                associated_hip = hips[wrist_idx]
                
                # Check if hand is holding an item
                held_item_id = active_tracks.get_held_item(pid, wrist_idx)
                
                if not held_item_id:
                    # Look for grabs
                    for item in detected_items:
                        if self.calculate_distance(wrist, item.center) < self.grab_threshold:
                            active_tracks.associate_grab(pid, wrist_idx, item.id)
                            break
                else:
                    # Person is holding an item. Check if it disappeared near pocket/hip
                    item_still_detected = any(item.id == held_item_id for item in detected_items)
                    
                    if not item_still_detected:
                        hip_dist = self.calculate_distance(wrist, associated_hip)
                        if hip_dist < pocket_threshold:
                            # Start disappearance countdown
                            active_tracks.mark_suspicious_disappearance(
                                pid, wrist_idx, held_item_id, target="pocket"
                            )
                            
            # 2. Check active disappearance counters
            completed_alerts = active_tracks.update_counters(self.confirm_frames)
            for alert in completed_alerts:
                alerts.append(alert)
                
        return alerts
```
