# Module Development Order - Phase 3 Production Planning

**Document Type**: Engineering Execution Sequence  
**Classification**: Implementation Guidelines  
**Audience**: Backend Developers, CV Engineers, QA Automation  

This document defines the exact order of implementation for all platform modules, detailing input-output contracts and verification strategies for each step.

---

## 1. Development Sequence

To ensure testing can be performed incrementally, the development sequence is organized as follows:

```text
Sequence:
[1] Edge SQLite DB ---> [2] Frame Decoder ---> [3] Detection (YOLO) ---> [4] Tracking (ByteTrack)
                                                                                  │
[8] Cloud DB <--- [7] API (FastAPI) <--- [6] Alert Engine <--- [5] Association & Behavior Engine
      │
      ▼
[9] React Dashboard ---> [10] End-to-End Edge & Cloud Deployment
```

---

## 2. Module Specifications

### Module 1: Edge Local Database Cache (`edge_cache.db`)
* **Why It Comes Now**: Serves as the configuration storage for camera stream URLs and coordinates. Downstream frame decoders need it to launch ingestion threads.
* **Required Inputs**: Local YAML configuration templates.
* **Expected Outputs**: SQLite database schema file; Python CRUD classes (`local_db.py`).
* **Verification / Test Method**: Unit tests verifying database creation, CRUD transactions of streams, and alert cache writes.
* **Integration Points**: Stream configurations are read by the Frame Decoder; alert cache is populated by the Alert Engine.

---

### Module 2: Frame Decoder & Ingestion Loop
* **Why It Comes Now**: Establishes the real-time frame buffer queue. All subsequent AI pipeline modules consume frames from this queue.
* **Required Inputs**: Stream configurations from Module 1 (IP Camera RTSP URLs).
* **Expected Outputs**: Shared memory frame queue (`/dev/shm`), raw frame numpy arrays with ingestion timestamps.
* **Verification / Test Method**: Replay test using a loopback RTSP video feed; measure frame rate consistency (15 FPS target) and memory usage profile over 24 hours.
* **Integration Points**: Frame queue is read by the Object Detector.

---

### Module 3: Object Detection (YOLOv12s Engine)
* **Why It Comes Now**: Extracts basic spatial data (bounding boxes and labels) from raw frames.
* **Required Inputs**: Frame queue from Module 2, YOLOv12s compiled weight files (ONNX/TensorRT).
* **Expected Outputs**: Structured metadata dictionary per frame: `[{class_label, bbox_coords, detection_confidence}]`.
* **Verification / Test Method**: Unit test verifying ONNX/TensorRT inference latency; regression test on 100 sample store images evaluating IoU bounds.
* **Integration Points**: Bounding boxes are read by the Tracker.

---

### Module 4: Multi-Object Tracking (ByteTrack Wrapper)
* **Why It Comes Now**: Converts single-frame bounding boxes into persistent spatial-temporal trajectories.
* **Required Inputs**: Bounding boxes and labels from Module 3.
* **Expected Outputs**: Metadata frame list augmented with tracking IDs: `[{track_id, class_label, bbox_coords}]`.
* **Verification / Test Method**: MOTA validation loop on standard MOT20 retail sequences, verifying identity switches remain low.
* **Integration Points**: Active track metadata is read by the Association Engine.

---

### Module 5: Association & Behavior Engine
* **Why It Comes Now**: Compiles the core spatial-temporal logic (Hungarian match wrist-to-item, pocket/bag overlap, loitering).
* **Required Inputs**: Tracked metadata from Module 4, pose keypoint arrays from YOLO-Pose, zone polygons from Edge config.
* **Expected Outputs**: Suspected behavior flags (`BehaviorFlag`), risk scores, and alert triggers.
* **Verification / Test Method**: **Mock coordinate testing**. We write JSON test files containing pre-defined hand/hip/item coordinate paths, feeding them directly to the logic engine to verify triggers without running actual YOLO video inference.
* **Integration Points**: Risk updates and event triggers are read by the Alert Engine.

---

### Module 6: Edge Alert Engine & Privacy Blurring
* **Why It Comes Now**: Encodes and anonymizes confirmed alert video clips locally, completing the edge-side processing loop.
* **Required Inputs**: Alert triggers from Module 5, local frame ring buffer from Module 2, pose head coordinates.
* **Expected Outputs**: Anonymized (Gaussian-blurred) 3-second MP4/GIF file.
* **Verification / Test Method**: Unit tests verifying face blur coordinate bounds; validation of video output compression constraints (<400KB).
* **Integration Points**: Dispatches alert package (JSON metadata + MP4 file) to the Cloud API Gateway.

---

### Module 7: Cloud Backend API (FastAPI)
* **Why It Comes Now**: Establishes the central ingestion platform for Edge nodes and Client browsers.
* **Required Inputs**: OpenAPI endpoint designs.
* **Expected Outputs**: FastAPI service stack; WebSocket connections, AWS S3 image-upload handlers.
* **Verification / Test Method**: Automated REST API tests using `pytest` and FastAPI's `TestClient`, checking auth, JSON validations, and mock S3 uploads.
* **Integration Points**: Connects to the Cloud PostgreSQL Database; communicates with the React Dashboard via WebSockets.

---

### Module 8: Cloud PostgreSQL Database Multi-Tenant Setup
* **Why It Comes Now**: Integrates the persistent storage backend for client analytics and logs.
* **Required Inputs**: PostgreSQL schema definition files.
* **Expected Outputs**: AWS RDS instance, initialized database tables and indexing patterns.
* **Verification / Test Method**: SQL migration integration scripts; query performance load testing under simulated high alert write volume.
* **Integration Points**: Managed by the Cloud API.

---

### Module 9: React Admin Dashboard
* **Why It Comes Now**: Provides the user interface to manage the platform.
* **Required Inputs**: Live API endpoints and S3 media URLs from Module 7.
* **Expected Outputs**: React/Vite web application build.
* **Verification / Test Method**: Frontend component tests (Cypress/Selenium); manual verification of WebSocket alert popups.
* **Integration Points**: Interacts with store managers via browser; communicates with FastAPI Backend.

---

### Module 10: End-to-End Edge & Cloud Deployment
* **Why It Comes Now**: Unifies all isolated modules into a single production-ready environment.
* **Required Inputs**: Edge Docker Compose files, Cloud Terraform/CloudFormation infrastructure scripts.
* **Expected Outputs**: Containerized Edge Gateway image stack, AWS ECS Fargate services, Active OTA agents.
* **Verification / Test Method**: End-to-end acceptance test (simulating a physical shoplifting event in a test store, checking if notification arrives on mobile within 5 seconds).
* **Integration Points**: Glues all physical hardware and SaaS platforms.
