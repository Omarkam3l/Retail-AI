# Model Selection and Evaluation - Phase 2 Design Review

This document serves as the official architecture reference for selecting, benchmarking, and configuring the artificial intelligence models for the Retail AI Surveillance Platform. 

---

## 1. Object Detection

### 1.1 Task Description
Detect and localize targets—specifically `person`, `backpack`, `handbag`, and generic products (`shelf_item`)—in real-time. Bounding boxes serve as inputs for the tracker and pose estimator.

### 1.2 Model Comparison

* **YOLOv11 (Ultralytics)**: CNN-based single-stage detector. Features C3k2 blocks and spatial attention.
* **YOLOv12 (Ultralytics)**: The latest iteration incorporating self-attention mechanisms directly into the backbone (using Area Attention modules) while preserving real-time CNN speed.
* **RT-DETR (Baidu)**: Real-Time DEtection TRansformer. Eliminates Non-Maximum Suppression (NMS) latency by using a transformer-based decoder.
* **Grounding DINO**: Open-vocabulary zero-shot detector. Extremely accurate at finding arbitrary text-described objects.

### 1.3 Benchmark Matrix (Target: RTX 3070, FP16)

| Model | mAP@50-95 (COCO) | Speed (Inference per Frame) | GPU Memory (VRAM) | Community Support | Ease of Deployment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11s** | 47.1% | ~4.2 ms | ~0.9 GB | Excellent (Large community) | Very Easy (ONNX / TensorRT) |
| **YOLOv12s** | **49.0%** | **~4.8 ms** | **~1.1 GB** | Growing (Ultralytics) | **Easy (TensorRT)** |
| **RT-DETR-R18** | 46.5% | ~6.5 ms | ~1.6 GB | Moderate (PaddlePaddle/HuggingFace)| Moderate (ONNX conversion complex) |
| **Grounding DINO**| 52.5% | ~90.0 ms | ~4.5 GB | Good (Research-focused) | Hard (Requires massive PyTorch runtime)|

### 1.4 Selection: **YOLOv12s** (Quantized to FP16)
* **Reason**: YOLOv12s offers the best balance of speed and localization accuracy. The introduction of attention layers in the backbone enables it to detect small, closely packed retail items on shelves far better than pure CNNs, while keeping latency under 5ms on the RTX 3070.
* **Trade-offs**: YOLOv12s is under the copyleft AGPL-3.0 license, which will require the startup to purchase a commercial license from Ultralytics for proprietary commercial distributions, or transition to RT-DETR (permissive Apache 2.0 license) if proprietary code isolation becomes a strict legal requirement.

---

## 2. Multi-Object Tracking (MOT)

### 2.1 Task Description
Maintain unique, consistent identities (Tracking IDs) for individuals moving through the store, associating detections across sequential frames under partial or complete occlusions.

### 2.2 Model Comparison
* **ByteTrack**: Motion-based tracker. Uses Kalman filtering and Hungarian matching. Uniquely matches low-score bounding boxes to recover occluded targets.
* **BoT-SORT**: Integrates Camera Motion Compensation (CMC) and deep Re-Identification (Re-ID) features into the association step.
* **DeepSORT**: Classical tracker combining Kalman filtering with a deep feature extraction network to compute appearance distances.
* **OC-SORT (Observation-Centric SORT)**: Focuses on tracking recovery during long occlusions by reconstructing trajectories backwards once targets reappear.

### 2.3 Benchmark Matrix (Target: Multi-Person tracking, 15 FPS)

| Tracker | IDF1 (Tracking Consistency) | Latency per Frame | GPU Memory (VRAM) | CPU Overhead | Ease of Deployment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ByteTrack** | 70.1% | **~0.9 ms** | **0 GB** (Runs on CPU/RAM) | **Very Low** | **Very Easy** (Pure Python/C++) |
| **BoT-SORT** | **74.5%** | ~7.2 ms | ~0.8 GB (Re-ID model) | High | Moderate |
| **DeepSORT** | 63.8% | ~9.5 ms | ~0.8 GB (Re-ID model) | High | Moderate (Requires custom CNN training) |
| **OC-SORT** | 72.3% | ~2.1 ms | 0 GB | Low | Easy |

### 2.4 Selection: **ByteTrack**
* **Reason**: ByteTrack is extremely fast and compute-light. Because it relies on IOU and motion geometry (Kalman filters) rather than extracting deep appearance features on every cropped bounding box, it consumes **0 GB of VRAM**. It is ideal for preserving GPU resources on the edge NVR.
* **Trade-offs**: Without a Re-ID appearance network, if a customer is fully occluded for more than 4 seconds (e.g. behind a wide pillar), ByteTrack will assign them a new tracking ID upon emergence, resetting their behavioral history. We mitigate this by tuning the Kalman filter keep-alive history to 90 frames (6 seconds).

---

## 3. Pose Estimation

### 3.1 Task Description
Track the 2D skeleton joints of shoppers to extract hand (`wrist`) and body (`shoulder`, `hip`) keypoints for gesture analysis.

### 3.2 Model Comparison
* **YOLOv11-Pose (Nano)**: Single-stage pose estimator. Outputs 17 keypoints directly in a single forward pass.
* **RTMPose (MMPose)**: High-performance real-time pose estimation framework designed for edge deployment.
* **OpenPose**: Multi-stage bottom-up keypoint detector. Resolves poses by computing Part Affinity Fields (PAFs).
* **ViTPose**: Vision Transformer-based pose estimator. Highly accurate but extremely heavy.

### 3.3 Benchmark Matrix (Target: Crop-based Multi-Person Inference)

| Model | mAP (Keypoints) | Latency per Crop | GPU Memory (VRAM) | Custom Training | Ease of Deployment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11-Pose (n)** | 61.5% | **~1.8 ms** | **~0.4 GB** | Easy | **Very Easy** (TensorRT native) |
| **RTMPose-S** | **68.2%** | ~2.9 ms | ~0.6 GB | Moderate | Moderate (ONNX / MMEngine runtime) |
| **OpenPose** | 58.4% | ~85.0 ms | ~3.5 GB | Hard | Hard (Legacy C++ dependencies) |
| **ViTPose-B** | 77.1% | ~45.0 ms | ~2.8 GB | Very Hard | Hard (Transformer edge runtime) |

### 3.4 Selection: **YOLOv11-Pose (Nano)** (Crop-Based)
* **Reason**: Sub-2ms latency per person crop. By only running inference on cropped bounding boxes of active tracks, we can process 10+ people simultaneously. It shares the same Ultralytics infrastructure as our detector, simplifying container builds.
* **Trade-offs**: Has slightly lower keypoint accuracy compared to RTMPose. However, the accuracy is more than sufficient to calculate the broad geometric relations needed for our behavior engine.

---

## 4. Human Action Recognition

### 4.1 Task Description
Classify and confirm suspicious sequences (grabbing and concealing an item).

### 4.2 Model Comparison
* **Rule-based Temporal Logic**: Evaluates Euclidean distances and bounding box intersections over a sliding time window.
* **ST-GCN (Spatio-Temporal Graph Convolutional Network)**: Classifies action categories directly from the skeleton keypoint trajectories over time.
* **Video Swin Transformer**: 3D video transformer model that processes raw spatio-temporal video volumes.
* **SlowFast (Facebook Research)**: Two-pathway 3D CNN (Slow path for spatial details, Fast path for temporal motions).
* **TimeSformer**: Processes video as a sequence of space-time patches using self-attention.

### 4.3 Benchmark Matrix (Target: 3-Second Action Sequence)

| Model | Accuracy (Gestures) | Latency / Frame | GPU Memory (VRAM) | Data Required | Deployment Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Heuristics** | Moderate (70-75%) | **<0.1 ms** (CPU) | **0 GB** | None (Code-defined)| **Selected for MVP** |
| **ST-GCN** | High (82-85%) | ~4.5 ms (GPU) | ~0.5 GB | High (Joint sequences)| **Selected for v2.0** |
| **SlowFast** | **Very High (88%)** | ~35.0 ms (GPU) | ~2.2 GB | Very High (Annotated MP4)| Rejected |
| **Video Swin / TimeSformer**| Very High (89%) | ~95.0 ms (GPU) | ~4.2 GB | Massive (Annotated MP4)| Rejected |

### 4.4 MVP vs. Future Version Strategy
* **MVP Selection**: **Rule-based Temporal Logic**.
  - *Why*: It executes instantly on the CPU, requires no expensive 3D convolutions, and has zero VRAM footprint. This leaves the RTX 3070 completely free to run the detector and tracker on multiple camera feeds.
  - *Trade-offs*: Vulnerable to false positives from normal gestures (e.g. scratching one's hip).
* **Future Selection**: **ST-GCN**.
  - *Why*: ST-GCN operates on coordinate graphs, not raw video frames. It consumes only ~0.5 GB of VRAM. It can learn complex, non-linear concealment signatures (e.g., natural pocketing speed vs. nervous/rapid pocketing) that are hard to code manually in rule heuristics.

---

## 5. Face Privacy

### 5.1 Task Description
Anonymize shoppers' faces prior to data transmission and storage to ensure strict compliance with GDPR and CCPA.

### 5.2 Model Comparison
* **Face Detection + Gaussian Blur (Selected)**: Locates the facial region (derived from YOLO-Pose keypoints `nose`, `eyes`, `ears`) and applies a local $31\times31$ Gaussian blur.
  - *Pros*: Extremely fast (<0.5ms), computationally simple, 100% GDPR compliant (completely destroys face pixel data).
  - *Cons*: Visually basic.
* **Segmentation Mask (Pixel replacement)**: Generates a pixel-accurate mask over the head/face and replaces it with solid color or a generic emoji.
  - *Pros*: Visually clean.
  - *Cons*: VRAM-heavy; adds significant latency.
* **Face Embeddings (Anonymization via hashing)**: Generates a vector representation of the face, replaces it in database, and discards raw video.
  - *Pros*: Allows cross-camera tracking without storing faces.
  - *Cons*: **Illegal in many jurisdictions**. Biometric profiling is heavily restricted; storing face vectors still constitutes holding PII.

---

## 6. Risk Scoring

### 6.1 Task Description
Evaluate multi-modal inputs (event types, zone flags, historical track scores) to generate a unified risk percentage.

### 6.2 Model Comparison
* **Rule Engine (Selected)**: Aggregates weights for specific actions:
  $$\text{Risk} = w_{pocket} \times S_{pocket} + w_{bag} \times S_{bag} + w_{loiter} \times S_{loiter}$$
  - *Pros*: 100% explainable, editable in real time via dashboard, execution is instantaneous.
* **Bayesian Network**: Models conditional probabilities of theft based on prior store layout statistics.
  - *Pros*: Good at handling missing data.
  - *Cons*: High configuration complexity.
* **ML Classifier (Random Forest/XGBoost)**: Trained on tabular logs of past alerts and associate feedback.
  - *Pros*: Automatically learns optimal weights.
  - *Cons*: Requires thousands of validated alert logs before becoming effective.
* **LLM-based Reasoning**: Passing event JSON summaries to a local LLM (e.g. Llama-3-8B) to output a risk assessment.
  - *Pros*: High contextual reasoning.
  - *Cons*: Massive VRAM requirements, extremely slow latency (>1.5s), unsuitable for real-time alerts.

---

## 7. Alert Generation & False-Positive Reduction

To prevent alerting fatigue, alerts pass through a multi-stage validation pipeline:

```text
┌───────────────────────┐
│ Behavior Detection    │ (Gesture flagged by Rules Engine)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Confidence Check      │ (Verifies item visibility & track score)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Cooldown Check        │ (Checks if tracking ID is in lockout phase)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Alert Dispatched      │ (WebSocket broadcast & FCM push notifications)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Human Confirmation    │ (Associate tags True/False Positive in App)
└───────────────────────┘
```

1. **Threshold-Based Adjustments**: Store managers can adjust behavior sensitivity (e.g. change loitering threshold from 120s to 240s, or modify pocket proximity range $\theta_{pocket}$ for specific cameras).
2. **Multi-Stage Verification**: An alert is only compiled if the tracking ID has a continuous state transition sequence (`Normal` -> `Interacting` -> `Suspicious` -> `Alert Triggered`). Single disconnected frames are discarded.
3. **Human Confirmation (Feedback loop)**: If an associate flags an alert as a "False Positive", the system automatically logs the coordinate trajectories and adds them as a negative training sample to prevent similar alarms.

---

## 8. Selected Pipeline Summary Table

| Stage | Selected Model | Reason | Trade-offs |
| :--- | :--- | :--- | :--- |
| **Object Detection** | **YOLOv12s (TensorRT)** | Built-in attention backbone improves detection of small retail items with sub-5ms latency. | AGPL-3.0 copyleft license; requires commercial license for proprietary distribution. |
| **Multi-Object Tracking** | **ByteTrack** | Fastest tracking; runs entirely on CPU/RAM with 0 VRAM footprint. | No appearance features; susceptible to ID loss during long occlusions. |
| **Pose Estimation** | **YOLOv11-Pose (Nano)** | Sub-2ms crop-based multi-person keypoint extraction. | Lower absolute mAP compared to heavy transformer networks. |
| **Action Recognition** | **Rule Heuristics** (MVP)<br>**ST-GCN** (v2.0) | Zero VRAM overhead, instant execution. ST-GCN handles skeletal temporal patterns in v2. | Vulnerable to false triggers from pocket-reaching motions in MVP. |
| **Face Privacy** | **YOLO-Pose Face BBox + Gaussian Blur** | Highly efficient, destroys all biometric PII, ensuring GDPR/CCPA compliance. | Basic visual appearance compared to segmentation masks. |
| **Risk Scoring** | **Weighted Rule Engine** | 100% explainable, low latency, easily configured by store owners. | Lacks dynamic context learning. |
| **Alert Generation** | **Multi-Stage Validation** | Combines state checks, cooldown locks, and human feedback to eliminate alert fatigue. | Relies on store associate feedback for optimization. |
