# Architecture Overview - Retail AI Surveillance Platform

This document describes the hybrid edge-cloud system architecture, camera feed ingestion, computer vision pipeline, and notification dispatch flow.

---

## 1. High-Level Architecture Diagram

Below is the logical flow of video frames, metadata, and alerts through the system:

```mermaid
graph TD
    %% Camera Ingestion (Edge)
    subgraph EdgeIngestion["1. Video Ingestion (Edge)"]
        CCTV["IP Cameras (CCTV)"] -->|RTSP / H.264| Dec["OpenCV Decoder / FFmpeg"]
        Dec -->|Raw Frames (15 FPS)| FrameBuf["Frame Queue Buffer"]
    end

    %% Computer Vision Pipeline (Edge)
    subgraph EVP["2. Edge Video Processing Pipeline"]
        FrameBuf -->|Retrieve Frames| ObjectDet["Object Detection (YOLOv8)"]
        ObjectDet -->|Bboxes: Person, Bag, Item| Tracker["Multi-Object Tracking (ByteTrack)"]
        Tracker -->|Tracked IDs + BBoxes| PoseEst["Pose Estimation (YOLO-Pose)"]
        PoseEst -->|Joint Keypoints| BehaviorEngine["Behavior Analysis Engine"]
    end

    %% Event & Alert Processing (Edge & Cloud)
    subgraph Alerts["3. Event & Alert Routing"]
        BehaviorEngine -->|Alert Triggered| EventDet["Event Detection & State Machine"]
        EventDet -->|Capture 3s Video Window| ClipEncoder["GIF/MP4 Clip Encoder"]
        ClipEncoder -->|Anonymize: Face Blur| PrivacyFilter["Privacy Filter"]
        PrivacyFilter -->|Post Meta + Clip| CloudGateway["Cloud Gateway (HTTPS)"]
    end

    %% Cloud Infrastructure
    subgraph CloudSaaS["4. Cloud Platform (SaaS)"]
        CloudGateway -->|Ingest Event| AlertService["Alert Dispatcher Service"]
        AlertService -->|Save Metadata| DB["PostgreSQL / SQLite Database"]
        AlertService -->|Save Video Clip| Storage["Cloud Object Storage (S3)"]
        
        AlertService -->|WebSocket Push| WebDash["Web Dashboard"]
        AlertService -->|FCM Push Notification| MobileApp["Mobile/Smartwatch App"]
    end

    %% Configuration Loop
    WebDash -->|Define Intrusion Polygons / Alerts| DB
    DB -->|Sync Camera Configs| BehaviorEngine
```

---

## 2. Module Descriptions

### 2.1 Camera Input (Ingestion)
* **Function**: Establish connections to local IP cameras using Real-Time Streaming Protocol (RTSP) streams.
* **Component**: Written in Python using OpenCV (`cv2.VideoCapture`) or a GStreamer pipeline.
* **Details**: Decodes compressed H.264/H.265 streams, downsizes frames to 640x640 pixels (standard YOLO inference size), skips frames to match a consistent 15 FPS target, and writes frames to a multi-threaded Queue.

### 2.2 Object Detection (YOLOv8)
* **Function**: Identify targets in each frame.
* **Component**: Ultralytics YOLOv8 (specifically optimized nano or small models like YOLOv8n/YOLOv8s).
* **Details**: Outputs bounding boxes, confidence scores, and class labels for `person`, `backpack`, `handbag`, and generic `shelf_item`.

### 2.3 Multi-Object Tracking (ByteTrack)
* **Function**: Maintain identities of people and luggage as they move.
* **Component**: ByteTrack (or SORT/DeepSORT wrapper).
* **Details**: Associates bounding boxes across frames using Kalman filters and Hungarian matching. Assigns a unique tracking ID (`ID_104`) to each customer.

### 2.4 Pose Estimation (YOLO-Pose)
* **Function**: Map the skeleton coordinates of shoppers.
* **Component**: YOLOv8-Pose.
* **Details**: Extract 17 skeleton keypoints, focusing on `left_wrist`, `right_wrist`, `left_shoulder`, `right_shoulder`, `left_hip`, and `right_hip`.

### 2.5 Behavior Analysis Engine
* **Function**: Analyze spatial and temporal relations.
* **Component**: Custom Rule & Heuristics engine.
* **Details**: Evaluates actions by analyzing:
  - **Intersection of Bounded Boxes**: Track if a `shelf_item` bounding box intersects with a customer's hand, and then disappears inside a `backpack` or a `person` (pocket region) bounding box.
  - **Trajectory Distance**: Calculate the spatial Euclidean distance between the hand keypoint (`wrist`) and the pocket keypoint (`hip`). A distance threshold $d < 10\text{ pixels}$ lasting for >10 frames suggests a concealment motion.

### 2.6 Event Detection & State Machine
* **Function**: Classify and validate alerts to prevent duplicate triggers.
* **Component**: Python-based state machine.
* **Details**: Moves through states: `Normal` -> `Interacting` -> `Suspicious (Concealment in progress)` -> `Alert Triggered`. Once in the `Alert Triggered` state, it locks additional alerts for that tracking ID for 60 seconds.

### 2.7 Privacy Filter (Anonymization)
* **Function**: Ensure GDPR compliance.
* **Component**: OpenCV image filter.
* **Details**: Detects the face region of the subject from the skeleton/keypoint coordinates and applies a strong Gaussian blur ($31\times31$ kernel) to the face *before* saving the video clip.

### 2.8 Alert Service & Clip Encoder
* **Function**: Encode, save, and route the event.
* **Component**: FFmpeg encoder + Python API client.
* **Details**: Captures 45 frames (15 frames pre-event, 30 frames post-event), compresses them into a lightweight looping MP4/GIF (< 400KB), and posts the payload to the Cloud Gateway via HTTPS.

### 2.9 Databases & Storage
* **Local Metadata**: SQLite database on the edge node to cache alerts during internet outages.
* **Cloud Database**: PostgreSQL for persistent records of stores, cameras, users, and alert histories.
* **Media Storage**: AWS S3 or MinIO cluster for hosting anonymized alert clips.

### 2.10 Dashboard & Client Apps
* **Web Dashboard**: Built in React/Vite. Allows managers to draw custom zone polygons (intrusion lines, high-shrink zones) and view analytics.
* **Mobile Client**: iOS/Android application displaying list of alerts, triggering vibration/alarms, and rendering the 3-second alert loops.

---

## 3. Data Flow Scenario (Theft Event)

1. **Detection**: User `ID_23` picks up a cosmetic bottle. YOLO detects `person` and `shelf_item`.
2. **Interaction**: User `ID_23` moves the bottle toward their jacket pocket.
3. **Trigger**: Pose estimation tracks the hand keypoint overlapping with the jacket pocket bounding box. The item disappears from the detector's view.
4. **Buffering**: The edge node extracts 3 seconds of video frames surrounding this movement.
5. **Blurring**: Faces of all individuals in those frames are blurred.
6. **Transmission**: The edge node encodes a 350KB MP4 clip and sends it to the Cloud Alert Service via HTTPS POST.
7. **Dispatch**: The Cloud Service stores the video in S3, logs the alert in PostgreSQL, and pushes the notification via WebSocket to the store's web dashboard and via FCM to the employees' mobile app.
8. **Resolution**: The store associate opens their phone, reviews the clip, confirms the concealment, and approaches the aisle.
