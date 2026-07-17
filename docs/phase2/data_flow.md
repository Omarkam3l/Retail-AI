# Data Flow Diagram - Phase 2 Design Review

This document illustrates the end-to-end data transmission path, detailing how video frames are ingested, processed, and transformed into lightweight alerts routed to edge caches, cloud storage, and client applications.

---

## 1. End-to-End Sequence Diagram

The sequence below outlines frame processing, event triggers, face blurring, cloud ingestion, and real-time alert dispatch.

```mermaid
sequenceDiagram
    autonumber
    participant Cam as CCTV IP Camera
    participant EdgeDec as Edge Decoder (OpenCV)
    participant EdgePipeline as Edge AI Pipeline
    participant EdgeEngine as Edge Behavior Engine
    participant CloudGateway as Cloud API Gateway
    participant CloudAlert as Cloud Alert Service
    participant CloudS3 as Cloud S3 Storage
    participant WebApp as React Dashboard / WebApp
    participant MobApp as Mobile / Watch App

    %% 1. Frame Ingestion
    Cam->>EdgeDec: Live RTSP Stream (H.264 packets)
    EdgeDec->>EdgeDec: Hardware Decode & Resize (640x640)
    
    %% 2. Processing Loop
    loop Every Frame (15 FPS)
        EdgeDec->>EdgePipeline: Raw Frame Buffer
        EdgePipeline->>EdgePipeline: Object Detection & ByteTrack Tracking
        EdgePipeline->>EdgePipeline: Crop-based Pose Estimation
        EdgePipeline->>EdgeEngine: Tracking IDs, BBoxes & Keypoints
    end

    %% 3. Suspicious Behavior Detection
    EdgeEngine->>EdgeEngine: Detect Concealment Gesture (State: Suspicious)
    Note over EdgeDec,EdgeEngine: Lock sliding video ring buffer (-30 frames)

    %% 4. Alert Triggered
    EdgeEngine->>EdgeEngine: Confirm Disappearance (State: Alert Triggered)
    Note over EdgeDec,EdgeEngine: Lock post-event video buffer (+15 frames)
    
    %% 5. Anonymization & Encoding
    EdgeEngine->>EdgePipeline: Request Face Blur on Frame Sequence (45 frames)
    EdgePipeline->>EdgePipeline: Apply Gaussian Blur on Face coordinates
    EdgePipeline->>EdgeEngine: Anonymized Video Frames
    EdgeEngine->>EdgeEngine: Encode Frame Sequence to Compressed MP4 (<400KB)

    %% 6. Cloud Transmission
    EdgeEngine->>CloudGateway: HTTPS POST /api/v1/alerts (Metadata JSON + MP4)
    
    %% 7. Cloud Orchestration
    CloudGateway->>CloudAlert: Route Event Payload
    par Storage & Logging
        CloudAlert->>CloudS3: Upload MP4 Clip
        CloudAlert->>CloudAlert: Save Event Log (PostgreSQL)
    end

    %% 8. Real-Time Push Notification Dispatch
    par Broadcast
        CloudAlert->>WebApp: WebSocket Broadcast (JSON Alert Metadata + S3 URL)
        CloudAlert->>MobApp: FCM Push Notification (Alert Notification + Payload)
    end

    %% 9. Associate Review
    MobApp->>CloudS3: Retrieve Anonymized Alert Clip
    MobApp->>MobApp: Play looping video clip to Employee
    MobApp->>CloudGateway: HTTP POST /api/v1/alerts/{id}/feedback (Confirm / Reject)
    CloudGateway->>CloudAlert: Log True/False Positive Feedback
```

---

## 2. Data Flow Steps Description

### 2.1 Frame Ingestion & Buffer Queue
1. The IP camera streams raw H.264 packets continuously over the local network via RTSP.
2. The Edge Ingestion thread decodes the H.264 packets using GPU hardware acceleration. The frame is downscaled to $640\times640$ pixels and timestamped.

### 2.2 Edge AI Pipeline Processing
3. Every frame in the ingestion buffer is parsed by YOLOv11n to extract object bounding boxes.
4. Bounding boxes are tracked by ByteTrack, maintaining consistent customer tracking IDs.
5. If tracking IDs intersect with high-shrink regions or display interaction, pose estimation is executed to extract body skeleton joints.
6. The compiled metadata (tracking IDs, coordinates, joints, labels) is forwarded to the Behavior Engine.

### 2.3 Gesture Logic & Buffer Locking
7. The Behavior Engine analyzes spatial distance rules. When a wrist keypoint is close to a pocket/bag and the item vanishes, the state machine transitions to `Suspicious`.
8. The system locks the rolling RAM ring buffer to preserve the preceding 30 frames (2.0 seconds) of video.

### 2.4 Confirmation & Edge Anonymization
9. If the item does not reappear after 30 frames, the state transitions to `Alert Triggered`. The succeeding 15 frames are captured.
10. The pipeline utilizes the skeleton head keypoints to map a facial bounding box on all 45 frames. A Gaussian Blur is applied locally.
11. The anonymized frames are compressed using H.264/WebP encoding into a lightweight clip file (<400KB).

### 2.5 Cloud Ingestion & Dispatch
12. The edge node sends the alert package (metadata JSON and binary video clip) to the Cloud API Gateway via HTTPS.
13. The API Gateway forwards the request to the Alert Service.
14. The Alert Service uploads the video file to an Amazon S3 bucket and records the transaction in the PostgreSQL database.
15. The Alert Service broadcasts the event payload over active WebSocket channels to the web dashboard and pushes an FCM notification to active android/iOS devices.

### 2.6 Feedback Loop
16. The store associate opens the notification, retrieves the S3 video URL, and reviews the looping clip.
17. The associate submits feedback ("True Shoplifting", "False Alarm", "Theft Prevented"), which is logged in the database to optimize model thresholds.
