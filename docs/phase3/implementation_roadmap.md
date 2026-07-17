# Implementation Roadmap - Phase 3 Production Planning

**Document Type**: Engineering Execution Plan  
**Classification**: Implementation Milestones  
**Audience**: Product Managers, Engineering Leads, Core Developers  

This document details the logical milestones to bring the Retail AI Surveillance Platform from design specifications to a production-grade deployment, focusing on risk mitigation, incremental testing, and minimal technical debt.

---

## 1. Milestones Overview

The project is structured into three main phases:
1. **Edge Pipeline Foundation (Milestones 1–5)**: Getting camera feeds decoded, running models, tracking entities, and associating products to hands.
2. **Behavior & Alerting Logic (Milestones 6–7)**: Building gesture rules, risk thresholds, and local event databases.
3. **Cloud & Client Infrastructure (Milestones 8–10)**: Building API services, websocket push endpoints, administrative dashboards, container packaging, and OTA agents.

```mermaid
gantt
    title Retail AI Surveillance - Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Edge Foundation
    M1: Video Ingestion & Decoding    :active, m1, 2026-08-01, 7d
    M2: Object Detection (YOLOv12s)   : m2, after m1, 10d
    M3: Multi-Object Tracking         : m3, after m2, 7d
    M4: Pose Estimation & Crop Loop   : m4, after m3, 10d
    M5: Object Association Engine     : m5, after m4, 14d
    section Behavior & Alerts
    M6: Behavior & State Engine       : m6, after m5, 14d
    M7: Risk Scoring & Local Cache    : m7, after m6, 10d
    section Cloud & Dash
    M8: Cloud API & Alert Dispatcher  : m8, after m7, 14d
    M9: React Admin Dashboard         : m9, after m8, 14d
    M10: Docker edge & AWS ECS Deploy : m10, after m9, 10d
```

---

## 2. Milestone Deep-Dive

### Milestone 1: Video Ingestion & Local Decoding Loop
* **Objective**: Establish robust, low-latency RTSP video packet collection and hardware-accelerated frame decoding.
* **Deliverables**:
  - `ingestion/rtsp_client.py` and `ingestion/decoder.py`.
  - Shared memory ring buffer (`/dev/shm`) allocating raw numpy arrays.
  - Multi-camera frame drop policy handler (drops frames if consumer thread lags).
* **Dependencies**: None.
* **Estimated Complexity**: Low (3 story points).
* **Exit Criteria**: Continuously ingests 4 RTSP 1080p camera feeds at 15 FPS on target edge hardware with 0% memory leakage and $<0.5\%$ frame drop rates over 48 hours.

---

### Milestone 2: Object Detection Integration
* **Objective**: Configure YOLOv12s, compile weights to TensorRT engines, and establish the frame processing queue consumer loop.
* **Deliverables**:
  - Export scripts for YOLOv12s ONNX and TensorRT engine compilation.
  - `pipeline/detector.py` implementing thread-safe inference.
  - Configuration pipeline validating detection labels (`person`, `backpack`, `handbag`, `shelf_item`).
* **Dependencies**: Milestone 1.
* **Estimated Complexity**: Medium (5 story points).
* **Exit Criteria**: Runs YOLOv12s TensorRT inference on 4 streams of 640x640 frame resolution at a stable $\geq 15\text{ FPS}$ with average latency $\leq 6\text{ms}$ per frame on RTX 3070.

---

### Milestone 3: Multi-Object Tracking State
* **Objective**: Implement ByteTrack to assign and maintain consistent customer tracking IDs.
* **Deliverables**:
  - `pipeline/tracker.py` wrapper around ByteTrack.
  - Trajectory centroid state memory queue.
  - Occlusion-recovering Kalman state predictors.
* **Dependencies**: Milestone 2.
* **Estimated Complexity**: Low (3 story points).
* **Exit Criteria**: Maintain active person track IDs across short-duration occlusions ($\leq 3\text{ seconds}$) with less than 2 identity switches per customer path in test footage.

---

### Milestone 4: Pose Estimation & Crop Loop
* **Objective**: Integrate crop-based YOLO-Pose Nano inference to retrieve body keypoint skeleton joint coordinates.
* **Deliverables**:
  - `pipeline/pose_estimator.py` supporting GPU-accelerated cropping of bounding box inputs.
  - Dynamic activation manager (turns pose detection on/off per track based on shelf proximity).
* **Dependencies**: Milestone 3.
* **Estimated Complexity**: High (8 story points).
* **Exit Criteria**: Pose keypoints successfully extracted on active customer crops in $<2.5\text{ms}$ per crop, with automatic pose deactivation when persons move away from shelves.

---

### Milestone 5: Object Association Engine
* **Objective**: Map relationship status between persons, hands, and products.
* **Deliverables**:
  - `behavior/association_engine.py` (Hungarian algorithm on wrists and product centers).
  - Product state lifecycle tracker (Unassociated, Candidate, Associated, Weak, Lost).
  - Disappearance classifier decision tree.
* **Dependencies**: Milestone 4.
* **Estimated Complexity**: Very High (13 story points).
* **Exit Criteria**: Correctly associates a picked shelf product with the specific picking hand in $\geq 92\%$ of test video sequences.

---

### Milestone 6: Behavior Engine & State Machine
* **Objective**: Implement the temporal state rules to detect pocket/bag concealment gestures.
* **Deliverables**:
  - `behavior/rules_engine.py` and `behavior/state_machine.py`.
  - Temporal Memory deque tracking track histories.
  - Coordinate templates for pocket and bag spatial constraints.
* **Dependencies**: Milestone 5.
* **Estimated Complexity**: High (8 story points).
* **Exit Criteria**: Concealment state successfully changes to `Alert Triggered` within 2 seconds of product disappearance near hip/bag.

---

### Milestone 7: Risk Scoring & Local Cache
* **Objective**: Aggregate temporal behavior flags, compute continuous risk scores, and write events to the local Edge cache database.
* **Deliverables**:
  - `controller/local_db.py` SQLite CRUD agent.
  - Weighted risk score calculator with linear decay functions.
  - Face-blurring OpenCV filter pipeline.
  - FFmpeg 45-frame loop MP4 encoder.
* **Dependencies**: Milestone 6.
* **Estimated Complexity**: Medium (5 story points).
* **Exit Criteria**: Compiles an anonymized (face-blurred) 3-second MP4 clip (<400KB) and saves it to local disk within 3 seconds of a confirmed concealment trigger.

---

### Milestone 8: Cloud API & Alert Dispatcher
* **Objective**: Deploy the central REST endpoints, S3 media management, database models, and WebSocket push broadcasters.
* **Deliverables**:
  - Cloud API server (FastAPI, SQLAlchemy, PostgreSQL schema).
  - `alert_service` handling Edge POST requests and uploading files to S3.
  - WebSocket connection hub routing real-time alerts.
* **Dependencies**: Milestone 7.
* **Estimated Complexity**: High (8 story points).
* **Exit Criteria**: End-to-end latency (from Edge POST to client WebSocket push receipt) is $\leq 200\text{ms}$ under a load of 500 requests/sec.

---

### Milestone 9: React Admin Dashboard
* **Objective**: Develop the management UI for video stream configuration, zone drawing, and real-time alert logs.
* **Deliverables**:
  - React/Vite client application.
  - Canvas-based interactive polygon drawer (restricted areas, shelves).
  - Looping video widget rendering anonymized S3 clips.
* **Dependencies**: Milestone 8.
* **Estimated Complexity**: Medium (5 story points).
* **Exit Criteria**: Displays incoming alerts in real-time without page refresh; allows saving zone coordinate polygons directly to cloud PostgreSQL.

---

### Milestone 10: Docker Edge & Cloud Deployment
* **Objective**: Package edge services into Docker Compose, deploy cloud resources, and configure OTA updating agents.
* **Deliverables**:
  - Edge `docker-compose.yml` and deployment installation scripts.
  - Cloud AWS ECS task configurations and S3 bucket IAM profiles.
  - Watchtower/Agent script managing edge node OTA containers.
* **Dependencies**: Milestone 9.
* **Estimated Complexity**: Medium (5 story points).
* **Exit Criteria**: Full system runs end-to-end across a remote edge gateway node and the cloud control plane, logging actions and dispatching mobile push notifications.
