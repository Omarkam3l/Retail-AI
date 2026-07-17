# Milestones & Verification Criteria - Phase 3 Production Planning

**Document Type**: Project Milestones  
**Classification**: Planning Reference  
**Audience**: Project Managers, Stakeholders, Dev Leads  

This document defines the key milestones for the development of the Retail AI Surveillance Platform and outlines the quantitative metrics used to verify the completion of each phase.

---

## 1. Milestone Roadmap

### Milestone 1: Video Ingestion & Decoding
* **Scope**: Ingestion thread, RTSP reader, VA-API/NVDEC decoder, frame shared memory queues.
* **Success Criteria**:
  - Stable connection and decoding of 4 distinct RTSP 1080p camera streams simultaneously.
  - Zero memory leaks over a 24-hour loop (RAM fluctuation $\leq 50\text{ MB}$).
  - Frame delivery rate to queue matches camera source (e.g. 15 FPS $\pm 0.5$ FPS).

### Milestone 2: Object Detection (YOLOv12s)
* **Scope**: YOLOv12s ONNX conversion, TensorRT engine compilation, inference loop.
* **Success Criteria**:
  - YOLOv12s inference latency $\leq 8\text{ms}$ on RTX 3070 for 640x640 resolution inputs.
  - Localization mAP@50 $\geq 90\%$ on target classes (`person`, `backpack`, `handbag`).
  - Core class confidence threshold calibrated (target: precision $\geq 85\%$ for active detections).

### Milestone 3: Multi-Object Tracking (ByteTrack)
* **Scope**: ByteTrack tracker class, Kalman filter, trajectory memory.
* **Success Criteria**:
  - Tracking latency $\leq 1.5\text{ms}$ per frame.
  - IDF1 tracking score $\geq 70\%$ on validation sequences.
  - Identity switches $\leq 3$ per 5-minute video loop of average density (5 shoppers).

### Milestone 4: Pose Estimation (YOLO-Pose)
* **Scope**: YOLO-Pose crop logic, wrist-hip keypoint extraction.
* **Success Criteria**:
  - Keypoint extraction latency $\leq 2.5\text{ms}$ per cropped person.
  - Object Keypoint Similarity (OKS) mAP@50 $\geq 75\%$ on wrist and hip joints.
  - Dynamic activation execution works: pose inference is only active when a track ID intersects with a shelf polygon.

### Milestone 5: Object Association Engine
* **Scope**: Hungarian matching algorithm, product state lifecycle tracks, disappearance decision tree.
* **Success Criteria**:
  - Association accuracy $\geq 92\%$ on test pickup video clips.
  - Disappearance classifier correctly isolates a pocket/bag concealment from a shelf occlusion in $\geq 88\%$ of cases.

### Milestone 6: Behavior Rules Engine & State Machine
* **Scope**: State transition logs, temporal gesture matching templates (concealment, loitering).
* **Success Criteria**:
  - 100% pass rate on the synthetic coordinate trajectory test suite (TDD test cases).
  - State machine changes customer state to `Alert Triggered` in $\leq 2.2$ seconds of a confirmed gesture sequence.

### Milestone 7: Risk Scoring & Edge Alerting
* **Scope**: Weighted scoring engine, Edge SQLite config cache, face-blurring filter, FFmpeg video loop encoder.
* **Success Criteria**:
  - Face blurring applied to head keypoints with zero raw facial exposures in output files.
  - Loop video compilation completes and writes file to disk in $\leq 3.0$ seconds of alert trigger.
  - Video loop file size $\leq 400\text{ KB}$ (average $250\text{ KB}$).

### Milestone 8: Cloud API & Ingestion Service
* **Scope**: FastAPI endpoints, JWT authorization, AWS S3 uploader, PostgreSQL schema database migrations, WebSocket alert broadcaster.
* **Success Criteria**:
  - REST API response time $\leq 150\text{ms}$ (p95) under a simulated concurrent load of 100 requests/sec.
  - Alert S3 upload and metadata PostgreSQL logging complete in $\leq 300\text{ms}$.
  - Real-time alert WebSocket broadcast received by connected clients within $50\text{ms}$ of API ingestion.

### Milestone 9: React Admin Dashboard
* **Scope**: Stream configuration panel, real-time alert toast, canvas polygon coordinate mapping tool.
* **Success Criteria**:
  - Canvas zone coordinates successfully drawn and saved to PostgreSQL.
  - Live alerts pop up on screen with video loop playback in $\leq 1.0$ second of Edge dispatch.

### Milestone 10: Edge-Cloud Integration & Deployment
* **Scope**: Edge Docker Compose stack, cloud AWS ECS deployment, OTA updater daemon.
* **Success Criteria**:
  - End-to-end alert notification latency $\leq 5.0$ seconds (from physical concealment in front of camera to smartphone push).
  - Edge node auto-recovers and restarts containers on power cycle/reboot within 90 seconds.
  - Config sync changes on dashboard are reflected on edge node SQLite database within 2.0 seconds.
