# AI Video Processing Pipeline - Phase 2 Design Design Review

This document defines the sequential stages of the edge-side computer vision pipeline, from raw video decoding to metadata extraction and privacy blurring.

---

## 1. Pipeline Cascade Overview

To process video in real-time (<65ms per frame) on low-power edge hardware, the system executes a **cascaded inference pipeline**. Instead of running all models on every pixel, we apply models sequentially, passing cropped regions of interest or using tracking history to bypass expensive computations.

```text
┌─────────────────┐
│ Live RTSP Feed  │
└────────┬────────┘
         │
         ▼
 ┌───────────────┐     No Persons Detected
 │ Object        ├─────────────────────────────┐
 │ Detection     │                             │
 └───────┬───────┘                             │
         │ Person Bboxes Detected              │
         ▼                                     │
 ┌───────────────┐                             │
 │ Multi-Object  │                             │
 │ Tracking      │                             │
 └───────┬───────┘                             │
         │ Active Tracks                       │
         ▼                                     │
 ┌───────────────┐                             │
 │ Pose          │                             │
 │ Estimation    │                             │
 └───────┬───────┘                             │
         │ Joint Coordinates                   │
         ▼                                     │
 ┌───────────────┐                             │
 │ Behavior      │                             │
 │ Rules Engine  │                             │
 └───────┬───────┘                             │
         │ Suspicious Flag Triggered           │
         ▼                                     │
 ┌───────────────┐                             │
 │ Face Blurring │                             │
 └───────┬───────┘                             │
         │ Anonymized Clip                     │
         ▼                                     │
 ┌───────────────┐     ┌───────────────────┐   │
 │ Clip Encoder  ├────>│   Cloud Upload    │   │
 └───────────────┘     └───────────────────┘   │
                               ▲               │
                               │ Normal Frame  │
                               └───────────────┴── Drop Frame / Skip
```

---

## 2. Pipeline Stages

### Stage 1: Video Ingestion & Decoding
* **Operation**: An independent ingestion thread reads live H.264 packets from the camera's RTSP stream.
* **Component**: OpenCV VideoCapture bound to hardware-accelerated decoders (NVIDIA NVDEC on Jetson, VA-API/QuickSync on Intel NUC).
* **Buffer Management**: Decoded frames are pushed into a bounded thread-safe Queue (capacity: 30 frames). If the queue is full (e.g., due to pipeline lag), the oldest frame is dropped to prevent lag and ensure real-time status.

### Stage 2: Object Detection (YOLOv8)
* **Operation**: Detects classes: `person`, `backpack`, `handbag`, `shopping_cart`, `shopping_basket`, and `shelf_item`.
* **Optimization**: 
  - Image scaling: Resize raw frame to $640\times640$ pixels preserving aspect ratio via letterboxing.
  - Frequency throttling: Object detection runs at 15 FPS. Intermediate frames (if camera is 30 FPS) are skipped or interpolated to save GPU cycles.

### Stage 3: Multi-Object Tracking (ByteTrack)
* **Operation**: Associates object detections across frames to maintain identity.
* **Mechanism**: ByteTrack parses the bounding boxes of `person` and carrying gear. It keeps track of the movement history (trajectory centroids) for each customer ID.
* **Optimization**: Bounding boxes with high confidence scores are matched first. Low-confidence boxes (which may be partially occluded) are matched secondarily using past trajectory directions, reducing identity switches.

### Stage 4: Person Crop & Pose Estimation (YOLO-Pose)
* **Operation**: Extracts 17-point joint skeleton coordinates for each active tracking ID.
* **Optimization (Crop-based Inference)**: 
  - To save massive processing power, we **do not run pose estimation on the entire frame**.
  - Instead, the frame is cropped around the bounding boxes of tracked `person` IDs.
  - The pose model is run only on these small, cropped regions.
  - The resulting relative keypoint coordinates are mapped back to the global frame coordinates.

### Stage 5: Behavior Rules Engine
* **Operation**: Evaluates spatial relations between the tracked `person`, `wrist` keypoints, carrying gear bounding boxes, and product bounding boxes.
* **Details**: Track wrist proximity to pockets or open bags. (Detailed rules are documented in `behavior_engine.md`).

### Stage 6: Privacy Blurring & Anonymization
* **Operation**: When the Behavior Engine or State Machine flags a suspicion state, a face-blurring filter is triggered.
* **Mechanism**:
  - The system utilizes the coordinates of the `nose`, `left_eye`, `right_eye`, `left_ear`, and `right_ear` keypoints from the pose estimation stage to compute the face bounding box region.
  - A Gaussian Blur filter with a kernel size of $31\times31$ is applied to this facial bounding box region in the frame buffer.
  - This guarantees that the suspect's face and any bystanders' faces are anonymized at the edge before storage or upload.

### Stage 7: Video Clip Compilation
* **Operation**: When an alert triggers, the edge node pulls the past 45 frames (3 seconds at 15 FPS) from a rolling memory ring buffer.
* **Encoding**: Compresses the frames into an H.264 MP4 file or a looping WebP animation, keeping file sizes under 400KB.
* **Routing**: Pushed to the cloud server via HTTPS POST.

---

## 3. Frame Synchronization & Time Stamp Matching

Because cameras can experience packet delays and variable frame rates, maintaining temporal alignment across multiple camera channels is critical.

* **Relative Epoch Timestamps**: Every decoded frame is immediately stamped at ingestion with its epoch timestamp (`t_ingest` in milliseconds) and its frame sequence index.
* **Timestamp Propagation**: This timestamp is attached to the frame metadata object and propagates through the detection, tracking, pose estimation, and behavior engines.
* **Dynamic Ring Buffer**: The edge node maintains a sliding window ring buffer of the last 150 frames (10 seconds) per camera. When the rules engine flags a theft event at timestamp $T$, the system can retrieve frames from $[T - 3000\text{ms}, T + 2000\text{ms}]$ using the metadata timestamps, ensuring that the compiled alert clip shows the precise moment of concealment.
* **Frame Skipping**: If the processing queue lags behind real-time by more than 3 frames (e.g., pipeline execution takes >200ms for a frame due to sudden density), the edge coordinator skips processing frames in the queue until it catches up, dropping frames but retaining tracking states via Kalman filter prediction updates.
