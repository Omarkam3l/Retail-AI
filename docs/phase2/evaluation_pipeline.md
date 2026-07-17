# Model Evaluation & Verification Pipeline - Phase 2 Design Review

This document defines the evaluation framework used to validate the accuracy, tracking stability, and real-time latency of our models before deployment, along with the active learning feedback loop.

---

## 1. Computer Vision Performance Metrics

To ensure robust performance, we evaluate each component of our AI pipeline independently using standardized computer vision benchmarks.

### 1.1 Object Detection (YOLOv11n)
* **Mean Average Precision (mAP@50-95)**: Evaluates the precision and recall trade-offs across 10 Intersection over Union (IoU) thresholds from 0.50 to 0.95.
* **Precision-Recall (PR) Curves**: Plotted specifically for critical target classes (`person`, `backpack`, `handbag`, `shelf_item`).
* **Target Thresholds**:
  - `person` class: mAP@50 >= 95%
  - `backpack` / `handbag` classes: mAP@50 >= 88%
  - `shelf_item` class: mAP@50 >= 78% (lower due to extreme diversity of retail products).

### 1.2 Multi-Object Tracking (ByteTrack)
* **MOTA (Multi-Object Tracking Accuracy)**: Evaluates false positives, missed targets, and identity switches.
* **IDF1 (Identification F1-Score)**: Measures the ratio of correctly identified detections to the average number of ground-truth and computed detections, highlighting track consistency over long sequences.
* **ID Switches (IDSW)**: Counts the frequency with which a tracked individual is assigned a new track ID (critical to keep low, as ID switches break the Behavior Engine's temporal rules).
* **Target Thresholds**:
  - MOTA >= 65% in retail aisle environments.
  - IDF1 >= 70%.

### 1.3 Pose Estimation (YOLO-Pose)
* **OKS (Object Keypoint Similarity) mAP**: Measures the accuracy of keypoint joint localization compared to ground truth, weighted by the specific joint type variance.
* **Target Thresholds**:
  - OKS mAP@50-95 >= 60% for wrist and hip keypoints.

---

## 2. System Performance & Latency Benchmarks

Models must be validated on actual target edge hardware (NVIDIA Jetson Orin Nano, Intel NUC Core i5) under simulated camera stream loads.

* **Latency Measurement Pipeline**:
  - We insert high-precision timers (`time.perf_counter_ns()`) around every execution step in the pipeline.
  - Latency metrics are compiled for: Ingestion (decode/resize), Detection Inference, Tracking Association, Pose Crop Inference, Rule Processing, and Face Blurring.
* **Frame Drop Rate (FDR)**: The percentage of video frames dropped due to queue congestion under continuous 4-camera execution. Target: **FDR < 0.5%**.

---

## 3. Human-in-the-Loop Active Learning Loop

Deploying models in the real world reveals edge cases (e.g. unique carrying containers, baggy jackets triggering false pocketing, sweeping motions to clear shelves). We implement a continuous feedback loop to ingest these edge cases and retrain models.

```text
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  ▼                                                        │
┌──────────────┐      Concealment Triggered      ┌─────────┴────┐
│ Store Edge   ├────────────────────────────────>│ Cloud SaaS   │
│ running AI   │                                 │ Database     │
└──────────────┘                                 └──────┬───────┘
  ▲                                                     │
  │ Retrained Model Engine                              │ Alerts Dashboard
  │ deployed via OTA                                    ▼
┌─┴────────────┐      Feedback logged in DB      ┌──────────────┐
│ Model        │<────────────────────────────────┤ Store Staff  │
│ Retraining   │ (True/False Positive tags)      │ (Mobile App) │
└──────────────┘                                 └──────────────┘
```

1. **User Feedback Collection**: Store associates classify every alert as:
   - **True Positive (TP)**: Theft or concealment occurred, or theft was prevented.
   - **False Positive (FP)**: Normal action flagged as theft (e.g. putting a phone in a pocket).
2. **Alert Archiving**:
   - For every **False Positive** alert, the 45-frame sequence, model bounding boxes, and joint keypoint coordinates are packaged and uploaded to a "Retraining Queue" in S3.
   - For every **False Negative (FN)** (theft reported by the manager that the system missed), the manager flag retrieves the corresponding raw NVR footage segment, which is labeled manually and added to the training set.
3. **Active Learning Curation**:
   - An automated script runs weekly in the cloud to extract these error clips.
   - High-difficulty clips are added to the fine-tuning training dataset, forcing the model to learn the boundary distinction between normal gestures (e.g. reaching for keys) and theft gestures.
4. **Nightly Regression Testing**:
   - Retrained models are tested against a golden validation test suite (1,000 baseline clips) to ensure accuracy does not degrade on old scenarios.
   - Successful models are compiled to TensorRT engines and deployed to edge nodes via OTA updates.
