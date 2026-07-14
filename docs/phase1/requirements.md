# System Requirements - Retail AI Surveillance Platform

This document outlines the functional requirements, non-functional requirements, technical constraints, deployment configurations, and security requirements for the Retail AI Surveillance Platform.

---

## 1. Functional Requirements (FR)

### 1.1 Video Ingestion & Stream Management
* **FR-1.1**: The system must ingest live H.264/H.265 video streams from existing IP cameras via RTSP or RTMP protocols.
* **FR-1.2**: The system must support camera resolution settings from 720p to 4K, running at variable frame rates (15 FPS to 30 FPS).
* **FR-1.3**: The system must handle network dropouts and automatically attempt to reconnect to camera streams without manual intervention.

### 1.2 Computer Vision & Inference Pipeline
* **FR-2.1**: The system must detect humans (customers/staff) and retail products with bounding boxes in real time.
* **FR-2.2**: The system must track individuals across frame boundaries within a single camera's field of view using a persistent tracking ID.
* **FR-2.3**: The system must map 2D skeleton pose keypoints (shoulders, elbows, wrists, hips) of tracked individuals.
* **FR-2.4**: The system must detect suspicious hand-to-pocket and hand-to-bag trajectories (concealment events) based on spatial overlaps and temporal sequences.

### 1.3 Alerts & Notifications
* **FR-3.1**: The system must compile alert events containing: Store ID, Camera ID, Timestamp, Event Type, and a 3-second compressed video clip (GIF/MP4).
* **FR-3.2**: The system must distribute real-time alerts to registered store staff via web dashboard (WebSockets) and mobile push notifications (FCM).
* **FR-3.3**: The system must allow users to classify alerts as "True Positive (Theft)", "True Positive (Prevented)", or "False Positive" to feed the active learning pipeline.

---

## 2. Non-Functional Requirements (NFR)

### 2.1 Latency & Speed
* **NFR-1.1**: The end-to-end alert latency (from the moment a concealment action is completed in front of the camera to the push notification arriving on the associate's device) must be **less than 5.0 seconds**.
* **NFR-1.2**: The video processing pipeline must process incoming frames at a minimum of **15 FPS** per camera to prevent frame lag and backlog.

### 2.2 Accuracy
* **NFR-2.1**: The concealment detection model must achieve a **Recall rate of >= 85%** under normal lighting conditions.
* **NFR-2.2**: The False Alarm Rate (FAR) must be kept below **1 false alert per camera per 8-hour shift** to prevent alert fatigue among store staff.

### 2.3 Reliability & Scalability
* **NFR-3.1**: The edge processor must run continuously (24/7) and support local storage buffers to cache video clips during internet outages.
* **NFR-3.2**: The cloud alert coordinator must scale horizontally to support up to 10,000 active retail stores.

---

## 3. Technical Constraints

* **Compute Constraints**: Edge deployment must run on budget-friendly compute devices. A single NVIDIA Jetson Orin Nano (8GB) or a standard Intel NUC (Core i5 with integrated graphics / OpenVINO) must be capable of processing up to **4 simultaneous 1080p camera feeds at 15 FPS**.
* **Bandwidth Constraints**: Small stores often have limited internet upload speeds (e.g., 5-10 Mbps). The system **must not upload continuous raw video streams** to the cloud. Only metadata (JSON) and short, highly compressed alert clips (average size < 500KB) may be sent over the WAN.
* **Framework Constraints**: The codebase must use Python 3.10+, PyTorch, and OpenCV. Model serialization should support ONNX or TensorRT/OpenVINO formats for hardware acceleration.

---

## 4. Deployment Requirements

The platform supports a **Hybrid Edge-Cloud Architecture**:

```text
┌──────────────────────────────────────┐          ┌──────────────────────┐
│             Store Edge               │          │      Cloud SaaS      │
│  ┌────────────┐      ┌────────────┐  │   WAN    │  ┌────────────────┐  │
│  │ IP Cameras │ ───> │ Edge Node  │  │ ───────> │  │ Alert Platform │  │
│  └────────────┘      └────────────┘  │  (HTTPS) │  └────────────────┘  │
└──────────────────────────────────────┘          └──────────────────────┘
```

### 4.1 Edge Node Software
* Running on Linux (Ubuntu 22.04 LTS).
* Containerized using Docker and Docker Compose for easy deployment, updates, and rollbacks.
* Operates a local HLS stream reader, YOLO inference, tracking, and local alerting buffer.

### 4.2 Cloud Control Plane
* Hosted on AWS or GCP.
* Handles customer account management, alert routing, notification dispatching, and aggregated analytics dashboards.
* Databases: PostgreSQL for transaction/alert metadata; Amazon S3 or Google Cloud Storage for evidence clips.

---

## 5. Security & Privacy Requirements

* **GDPR & CCPA Compliance (Face Blurring)**: To comply with privacy laws, the edge node **must apply a real-time Gaussian blur to human faces** before any alert clips are saved locally or uploaded to the cloud. No facial biometric templates may be generated or stored.
* **Encryption**: All data transmitted between the store edge node and the cloud servers must be encrypted using TLS 1.3. Local video clips stored on the edge device must be encrypted using AES-256.
* **Access Control**: Role-Based Access Control (RBAC) must distinguish between "Store Associate" (can view real-time alerts on mobile) and "Store Manager" (can modify camera regions, view analytics, and export evidence clips).
