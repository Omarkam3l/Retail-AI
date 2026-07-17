# Risks & Limitations - Phase 2 Design Review

This document identifies the technical risks, physical limitations, and compliance challenges associated with the deployment of the Retail AI Surveillance Platform, providing mitigation strategies for each.

---

## 1. Technical & Environment Risks

### 1.1 Dense Crowds & Occlusions
* **Risk**: Multiple customers walking closely together, crossing paths, or standing in front of shelves blocks the camera line of sight (occlusion). This causes:
  - Object detection failure (targets are partially hidden).
  - Tracking ID switches (ByteTrack assigns the wrong ID to a person emerging from behind another).
  - Broken behavior rules (the engine misses the grab or concealment sequence).
* **Mitigation**:
  - Integrate **OC-SORT** (Observation-Centric SORT) or Kalman filters with momentum vector calculation to predict the trajectory of a person during occlusion based on their entry speed and direction.
  - Direct store managers to install cameras at high angles ($30^\circ$ to $45^\circ$ pitch) to minimize overlapping silhouettes.

### 1.2 Poor Lighting Conditions
* **Risk**: Dimly lit aisles, shadows, or glare from glass-front refrigerators reduce the contrast of camera feeds, causing high detection error rates.
* **Mitigation**:
  - Implement basic frame preprocessing on the edge node (e.g. CLAHE - Contrast Limited Adaptive Histogram Equalization) before passing frames to YOLO.
  - Leverage cameras' built-in infrared (IR) night-vision modes; validate models on IR grayscale datasets to ensure tracking and keypoint detection accuracy persists in monochrome.

### 1.3 False Positives from Normal Actions
* **Risk**: Customers frequently perform actions that resemble concealment, such as putting their hands in their pockets (reaching for keys, wallet, or phone) or placing personal items into their own shopping bags.
* **Mitigation**:
  - **Temporal Grace Period**: Require the item to disappear for a minimum of 30 frames (2 seconds) before triggering an alert.
  - **Basket Exclusions**: Detect store-specific baskets. If an item disappears inside a store basket (classified by color/shape heuristics), suppress the alert.
  - **Proactive Customer Service Pitch**: Instruct store staff to treat alerts as "customer service opportunities" rather than accusations (e.g. approaching a loitering or suspicious customer to say, "Can I get a shopping basket for you?"). This defuses false alarms while deterring actual shoplifters.

---

## 2. Infrastructure & Hardware Risks

### 2.1 Network Downtime & Bandwidth Constraints
* **Risk**: Store broadband connection drops, preventing edge nodes from uploading alert clips or receiving config updates.
* **Mitigation**:
  - **Local SQLite Caching**: The edge node caches alerts in the local SQLite database and saves the compressed video files to local disk storage.
  - **Background Resync**: A background sync worker continuously polls connection health. When internet connectivity is restored, cached alerts are uploaded sequentially in the background, throttling upload speed to avoid saturating store bandwidth.

### 2.2 Edge Hardware Failures
* **Risk**: Local edge computers (Jetson, NUC) freeze, overheat, or experience power failures, leaving the store unprotected.
* **Mitigation**:
  - **Systemd Watchdogs**: A system-level watchdog script monitors the health of Docker containers. If the memory queue blocks or a service crashes, systemd automatically restarts the containers.
  - **Heartbeat Monitoring**: The edge node sends a ping heartbeat (every 30 seconds) to the Cloud Control Plane. If pings cease for >3 minutes, the cloud sends an email/push notification to the store manager: *"Warning: Edge Node for Store X is offline."*

---

## 3. Regulatory & Privacy Compliance (GDPR/CCPA)

* **Risk**: Storing or transmitting videos of citizens in public areas risks violating strict privacy laws.
* **Mitigation**:
  - **Edge Face Blurring**: The edge node detects and applies a Gaussian blur to all faces before saving any media clip locally or uploading it to the cloud.
  - **No Biometric Matching**: The platform does not use facial recognition, database face matching, or individual biometric profiling.
  - **Data Retention Policies**: Cloud-stored evidence clips are automatically deleted after 30 days unless marked as "active legal case" by store managers.
