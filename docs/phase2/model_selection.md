# Model Selection & Benchmarking - Phase 2 Design Review

This document evaluates and selects the optimal machine learning models for object detection, multi-object tracking, and pose estimation. 

---

## 1. Object Detection Model Comparison

The object detector must identify `person`, `backpack`, `handbag`, and generic products (`shelf_item`) in real-time under edge hardware constraints. We compare the nano (n) and small (s) variants of YOLOv8, YOLOv10, and YOLOv11.

| Model Variant | Params (M) | mAP @50-95 (COCO) | Latency (Jetson Orin Nano - FP16) | License | Selection Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | 3.2M | 37.3% | ~6.5 ms | AGPL-3.0 | Evaluated |
| **YOLOv8s** | 11.2M | 44.9% | ~12.8 ms | AGPL-3.0 | Evaluated |
| **YOLOv10n** | 2.3M | 38.5% | ~5.8 ms | AGPL-3.0 | Evaluated |
| **YOLOv10s** | 7.2M | 46.3% | ~11.2 ms | AGPL-3.0 | Evaluated |
| **YOLOv11n** | 2.6M | 39.4% | ~5.2 ms | AGPL-3.0 | **Selected (Base Detector)** |
| **YOLOv11s** | 9.4M | 47.1% | ~9.8 ms | AGPL-3.0 | Evaluated (Future high-end option) |

### Architectural Decision & Trade-offs
* **Selected**: **YOLOv11n** (quantized to FP16/INT8 via TensorRT).
* **Rationale**: YOLOv11n offers the best accuracy-to-latency ratio. Its architectural improvements (enhanced C3k2 blocks and spatial attention) provide superior detection of small objects (like retail goods) compared to YOLOv8n, while consuming fewer parameters.
* **Trade-off (License)**: Ultralytics models are licensed under AGPL-3.0, which requires open-sourcing modifications or purchasing a commercial license. For the MVP, we use the open-source version, planning a commercial license transition or switching to a permissive model (e.g., RT-DETR or YOLOv10 with Apache 2.0 license) if strict commercial IP separation is required later.

---

## 2. Multi-Object Tracking (MOT) Comparison

A tracking framework must associate detected bounding boxes across sequential video frames, maintaining persistent customer IDs under frequent occlusions.

| Tracker Algorithm | Re-ID Model Needed | MOTA | Latency per Frame | Strengths | Weaknesses | Selection Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SORT** | No | 55.4% | ~0.8 ms | Extremely fast | High identity switches under occlusions | Rejected |
| **DeepSORT** | Yes (Feature extractor) | 61.2% | ~8.5 ms | Strong Re-ID matching | High CPU overhead for feature extraction | Rejected |
| **ByteTrack** | No (IOU + Kalman) | **68.2%** | **~1.2 ms** | Matches low-score boxes; very fast | Can lose track in extreme crowd density | **Selected (MVP)** |
| **OC-SORT** | No (Direction-aware) | 69.8% | ~2.5 ms | Excellent handling of camera noise/stops | Higher compute overhead than ByteTrack | Future Option |

### Architectural Decision & Trade-offs
* **Selected**: **ByteTrack**.
* **Rationale**: ByteTrack utilizes a simple, highly efficient data association method. Unlike DeepSORT, which runs an expensive deep feature extractor on every bounding box, ByteTrack uses Kalman filtering and IOU matching. It uniquely matches low-confidence bounding boxes (e.g., when a person is partially occluded by a shelf), significantly reducing identity switches without extra GPU overhead.

---

## 3. Pose Estimation Model Comparison

Pose estimation is required to track the customer's wrists relative to pockets, bags, and shelves.

| Pose Estimation Model | Keypoints | Latency per Person (Crop-based) | Resource Profile | Accuracy (mAP Keypoints) | Selection Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MediaPipe Pose** | 33 | ~15 ms (CPU-bound) | Heavy CPU / Light GPU | Very high | Rejected (Hard to scale multi-person) |
| **YOLOv8-Pose (Nano)** | 17 | ~2.2 ms (GPU TensorRT) | Light GPU | High (59.2 mAP) | **Selected (MVP)** |
| **YOLOv11-Pose (Nano)** | 17 | ~1.9 ms (GPU TensorRT) | Light GPU | High (61.5 mAP) | Selected (Secondary Option) |

### Architectural Decision & Trade-offs
* **Selected**: **YOLOv8-Pose (Nano)** / **YOLOv11-Pose (Nano)**.
* **Rationale**: YOLO-Pose is a single-stage keypoint estimator. By running it on cropped bounding boxes of active tracks, we achieve sub-2ms latency per person. MediaPipe Pose is optimized for single-person mobile applications and does not scale efficiently to 10+ shoppers in a store aisle.
* **Trade-off**: YOLO-Pose has fewer keypoints (17) compared to MediaPipe (33), but it includes all the major joints (wrists, elbows, hips, shoulders) needed for behavior heuristics.

---

## 4. Edge Hardware Resource Budget (NVIDIA Jetson Orin Nano 8GB)

To fit our SME cost constraints, we budget our models to run on a single $300 edge gateway:

* **Hardware Specs**: 40 TOPS AI compute, 6-core ARM CPU, 8GB shared LPDDR5 memory (shared between CPU and GPU).
* **VRAM Allocation Budget**:
  - Operating System + Drivers: **1.5 GB**
  - OpenCV Video Decoders + Frame Buffers (4 channels @ 1080p): **1.2 GB**
  - YOLOv11n Object Detector (TensorRT FP16): **0.8 GB**
  - YOLOv8-Pose Nano (TensorRT FP16): **0.6 GB**
  - Edge Cache & Tracker State: **0.5 GB**
  - **Remaining Headroom / Safety Margin**: **3.4 GB** (ensures system stability).
* **Latency Budget (Target: <65ms per frame loop)**:
  - Video Decoding & Letterboxing: **8 ms**
  - YOLOv11n Detection: **6 ms**
  - ByteTrack Association: **1.5 ms**
  - YOLO-Pose Keypoints (on 2 active tracks): **4 ms**
  - Heuristics Rule Processing: **2 ms**
  - **Total Loop Latency**: **21.5 ms** (comfortably supports 15 FPS / 66ms budget per stream for up to 4 cameras simultaneously).
