# System Architecture - Phase 2 Design Design Review

This document defines the high-level system architecture of the Retail AI Surveillance Platform, including the logical partition between Edge and Cloud, store-level physical deployment, and architectural trade-offs.

---

## 1. High-Level Logical Architecture

The platform uses a **Hybrid Edge-Cloud Architecture**. Heavy, latency-critical, privacy-sensitive workloads (video ingestion, decoding, object tracking, pose estimation, and face blurring) run locally on store-level edge nodes. Lightweight coordination, persistent storage, and alerting routing run on multi-tenant cloud infrastructure.

```mermaid
graph TB
    subgraph Store_Edge["Store Edge (Local Network)"]
        Cameras["IP Cameras (CCTV Feeds)"] -->|RTSP / H.264| EdgeNode["Edge Gateway Node (Docker Stack)"]
        
        subgraph EdgeNodeComponents["Edge Gateway Containers"]
            Ingestion["Ingestion & Decoding Service"] -->|Raw Frame Queue| CV["Computer Vision Pipeline"]
            CV -->|BBoxes & Keypoints| RulesEngine["Behavior Rules Engine"]
            RulesEngine -->|Triggered Events| StateMachine["Alert State Machine"]
            StateMachine -->|Event Frame Sequence| BlurService["Privacy Face Blur Filter"]
            BlurService -->|Looping MP4/GIF Encoder| ClipEncoder["Clip Encoder"]
            
            DBLocal[("Edge SQLite Cache")] <--> Ingestion & CV & StateMachine
        end
    end

    subgraph Cloud_Backend["Cloud Control Plane (SaaS)"]
        CloudGateway["API Gateway / Load Balancer"]
        
        subgraph CloudServices["Microservices"]
            AlertService["Alert Orchestrator"]
            ConfigService["Config Synchronizer"]
            AnalyticsService["Analytics Engine"]
        end
        
        DBCloud[("PostgreSQL Multi-tenant")]
        S3Storage[("AWS S3 (Anonymized Clips)")]
    end

    subgraph Client_App["Client Presentation Layer"]
        Dashboard["React Web Dashboard (Managers)"]
        MobileApp["Mobile / Smartwatch App (Associates)"]
    end

    %% Network Connections
    ClipEncoder -->|HTTP POST: Metadata + Clip| CloudGateway
    StateMachine -->|HTTP POST: Metadata| CloudGateway
    CloudGateway <--> AlertService & ConfigService & AnalyticsService
    
    AlertService -->|WebSockets / Server-Sent Events| Dashboard
    AlertService -->|FCM Push Notification| MobileApp
    ConfigService -->|WebSocket Config Sync| EdgeNode
    
    AlertService -->|Store Metadata| DBCloud
    AlertService -->|Upload Video| S3Storage
    Dashboard & MobileApp -->|Query Analytics & Configs| CloudGateway
```

---

## 2. Store-Level Physical Topology

SMEs typically possess standard analog or IP cameras wired to a Network Video Recorder (NVR) or connected to a local PoE (Power-over-Ethernet) switch. 

* **IP CCTV Cameras**: Connect to a local router/PoE switch. Feeds are exposed locally via RTSP URLs (e.g., `rtsp://admin:password@192.168.1.50:554/h264`).
* **Edge Gateway Node**: A physical compact computer (e.g., NVIDIA Jetson Orin Nano, Intel NUC, or mini-PC) connected to the same local network subnet. It accesses RTSP feeds, processes them, and connects to the internet via the store’s router.
* **Store Router**: Connects to the ISP WAN. Only handles outgoing HTTPS posts for alerts and configuration WebSocket polls (minimal bandwidth).

---

## 3. Edge vs. Cloud Responsibilities

| Responsibility / Feature | Executed On | Rationale |
| :--- | :--- | :--- |
| **Video Decoding & Frame Extraction** | **Edge** | Ingesting high-definition video over the internet is bandwidth-prohibitive. |
| **AI Model Inference (YOLO / Pose)** | **Edge** | Running deep learning on continuous video streams in the cloud incurs massive hosting fees. Edge processing ensures <65ms processing latency. |
| **Face Blurring (GDPR compliance)** | **Edge** | Biometric and identifiable data must never leave the store's physical premises. |
| **Alert State Machine / Heuristics** | **Edge** | Evaluates gestures locally to trigger video buffering immediately. |
| **Alert Routing & Push Dispatch** | **Cloud** | Orchestrates mobile push notifications (FCM/APNS) and maintains socket connections. |
| **Persistent Evidence Storage** | **Cloud** | Local edge drives can be stolen, damaged, or run out of disk space. Cloud storage (S3) secures evidence. |
| **Store Management Dashboard** | **Cloud** | Allows store managers to view analytics and modify configurations remotely. |

---

## 4. Architectural Trade-offs

During the design phase, three deployment topologies were evaluated:

### Option A: Cloud-Only Processing
Live video streams are uploaded to a cloud server cluster for central inference.
* *Pros*: Zero hardware cost for the retailer; easy model deployment and updates.
* *Cons*: Requires massive internet bandwidth (e.g., 5-10 Mbps upload *per camera*), leading to high network congestion and retail downtime. Incurs extremely high cloud GPU bills.
* *Decision*: **Rejected**. Unviable for SMEs with typical retail internet connections.

### Option B: Edge-Only Processing
All video processing, alerting, dashboard serving, and database storage are handled locally by a high-spec on-premise server.
* *Pros*: Zero cloud subscription dependency; fully offline-functional.
* *Cons*: High upfront CapEx for the merchant (needs a powerful PC with RTX GPU). Vulnerable to data loss if the local server is damaged or stolen. No remote access for store managers.
* *Decision*: **Rejected**. CapEx and management complexity are too high for SMEs.

### Option C: Hybrid Edge-Cloud (Selected)
Lightweight edge nodes process streams locally and send only compressed alert metadata and short anonymized clips to the cloud control plane.
* *Pros*: Lowest bandwidth footprint (no raw stream uploads), low cloud GPU costs, low edge hardware CapEx (supports Jetson/low-spec NUC), secure cloud backups, and remote management.
* *Cons*: Requires managing Docker deployments on distributed edge hardware.
* *Decision*: **Selected**. Provides the optimal balance between SME CapEx, WAN network constraints, and SaaS operating expenses.
