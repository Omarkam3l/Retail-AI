# Testing Strategy & QA Plan - Phase 3 Production Planning

**Document Type**: Quality Assurance Specification  
**Classification**: Test Design Reference  
**Audience**: QA Engineers, CV Developers, CI/CD Managers  

This document defines the testing strategy for the Retail AI Surveillance Platform, ensuring accuracy, latency bounds, and reliability across edge and cloud layers.

---

## 1. Test Levels & Methods

We implement a layered testing strategy to validate every component of the hybrid architecture.

```text
┌─────────────────────────────────────────────────────────┐
│                   Acceptance Tests                      │ (End-to-End theft alert loop)
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│     Performance, Stress & Video Regression Tests       │ (RTX 3070 frame rate, golden video sets)
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│         Replay, Synthetic & Integration Tests           │ (Replaying NVR loops, JSON coordinates)
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                   Unit Tests (TDD)                      │ (Rules logic, database CRUD)
└─────────────────────────────────────────────────────────┘
```

### 1.1 Unit Tests (TDD Focused)
* **Scope**: Individual modules, utility functions, SQLite/PostgreSQL models, and behavior heuristics.
* **TDD Principle (Write Before Implement)**: For the **Behavior Rules Engine** (`rules_engine.py`) and **State Machine** (`state_machine.py`), unit tests **must be written before implementation code**. We write test fixtures representing customer coordinates (e.g., wrist coordinates moving from shelf to pocket). The behavior code is then written to satisfy these coordinate assertions.
* **Framework**: `pytest`.

### 1.2 Synthetic Coordinate Tests
* **Scope**: Rules validation without running GPU models.
* **Method**: Write script generators that compile simulated Coordinate trajectory matrices representing:
  - Perfect pocketing gesture.
  - Pocket reach with no product.
  - Handover between two people.
  - Rapid exit.
* **Why it's used**: Allows testing 100+ edge-case gesture variations in milliseconds on standard CPU pipelines.

### 1.3 Replay Tests (Local Edge Pipeline)
* **Scope**: Edge node stream decoding and YOLO ingestion loop.
* **Method**: A mock RTSP loopback server (using GStreamer or RTSP-Simple-Server) replays pre-recorded raw video footage of retail store aisles. The local Edge node connects to this mock stream, verifying H.264 stream reading, decoding, queue populating, and tracking stability.

### 1.4 Video Regression Tests
* **Scope**: Protecting model accuracy against regressions.
* **Method**: A "Golden Video Validation Set" (100 video clips of actual shoplifting, simulated theft, and standard shopping actions) is maintained in S3. 
* **CI/CD Integration**: When a developer updates YOLO weights, tracking hyperparameters (e.g. ByteTrack IOU match threshold), or behavior rules, the Golden Video set is processed automatically. The system asserts that the detection rate does not decline, and false alerts do not increase compared to baseline metrics.

### 1.5 Performance & Memory Tests
* **Scope**: Latency bounds and memory allocation profiles.
* **Metrics Tracked**:
  - Processing latency per frame (Target: $\leq 65\text{ms}$).
  - Edge memory leak checking (VRAM and RAM allocation stability over 48 hours).
  - CPU thread context-switch overhead (ingestion thread vs. inference thread).

### 1.6 Stress & Crowd Scale Tests
* **Scope**: System performance under peak load (e.g., Black Friday).
* **Method**: Ingest video loops containing extremely crowded aisles (20+ active tracking targets). Verify:
  - Edge queue capacity limits (does the frame drop policy execute gracefully?).
  - Memory consumption limits (VRAM remains within the 8GB RTX 3070 boundaries).
  - ByteTrack Kalman filter tracking latency under high identity counts.

### 1.7 End-to-End Acceptance Tests
* **Scope**: The complete data path from physical camera to mobile user.
* **Exit Criteria**: A physical actor in a test store conducts a mock pocketing gesture. The mobile app must trigger a vibration alert with the blurred video clip in **less than 5.0 seconds**.

---

## 2. Test Execution Matrix

| Test Category | Trigger Frequency | Environment | Target Components | Success Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Unit Tests** | Every git commit | GitHub Actions runner | Database, State Machine, API controllers | 100% pass rate, $\geq 80\%$ code coverage |
| **Synthetic Tests**| Every git commit | GitHub Actions runner | Behavior Engine, Risk Matrix | 100% pass rate |
| **Replay Tests** | Weekly | Local Edge Staging Node | Video Decoder, Object Tracker | Stable 15 FPS, $\leq 0.5\%$ frame drop rates |
| **Video Regression**| Pull Request merge | GPU-equipped CI runner | YOLOv12s, Tracker, Pose, Rules | Recall $\geq 85\%$, False Alarm $\leq 1$/camera/shift |
| **Stress Tests** | Prior to release | Local Edge Staging Node | Queue, Track Memory, Frame Dropper | No container crashes, VRAM $\leq 6\text{ GB}$ |
| **E2E Acceptance** | Prior to release | Physical Staging Store | Edge Node, Cloud SaaS, Web/Mobile App | Alert latency $\leq 5\text{s}$ (p90) |
