# Deployment Architecture - Phase 2 Design Review

This document describes the containerized edge deployment, cloud infrastructure orchestration, bandwidth management mechanisms, and Over-The-Air (OTA) update procedures.

---

## 1. Edge Node Containerization (Docker Compose)

The edge gateway node runs a lightweight Ubuntu 22.04 LTS OS. All application components are packaged as Docker containers, orchestrated locally via Docker Compose.

```yaml
version: '3.8'

services:
  # Ingests video streams, decodes H.264, and populates shared memory frame queue
  video-ingestion:
    image: retail-ai/video-ingestion:latest
    container_name: video-ingestion
    network_mode: host
    restart: unless-stopped
    devices:
      - /dev/nvidia0:/dev/nvidia0 # Access to Jetson GPU
    volumes:
      - shared-frame-buffer:/dev/shm # IPC shared memory for frames
      - ./config/streams.json:/app/config/streams.json
    logging:
      driver: "json-file"
      options:
        max-size: "10m"

  # Core AI pipeline: runs object detection, tracking, and pose estimation
  ai-inference-engine:
    image: retail-ai/inference-engine:latest
    container_name: ai-inference-engine
    network_mode: host
    restart: unless-stopped
    runtime: nvidia # NVIDIA Container Toolkit integration
    environment:
      - USE_CUDA=true
    volumes:
      - shared-frame-buffer:/dev/shm
      - ./models:/app/models
    depends_on:
      - video-ingestion

  # Local DB cache and event analyzer (Behavior + State Machine)
  edge-behavior-controller:
    image: retail-ai/behavior-controller:latest
    container_name: edge-behavior-controller
    network_mode: host
    restart: unless-stopped
    volumes:
      - shared-frame-buffer:/dev/shm
      - ./data:/app/data # SQLite database persistence
    environment:
      - API_GATEWAY_URL=https://api.retail-ai.com
      - EDGE_AUTH_TOKEN=7db8e990-bc10-44ff-a002-deefaa334455
    logging:
      driver: "json-file"
      options:
        max-size: "10m"

volumes:
  shared-frame-buffer:
    driver: local
```

---

## 2. Cloud Orchestration (AWS ECS / Fargate)

To keep maintenance overhead minimal for a startup, the cloud backend utilizes a serverless container orchestration model (AWS ECS Fargate) rather than managing a complex Kubernetes (EKS) cluster for the MVP.

* **ECS Fargate Task Definition**: Stateless FastAPI instances run as individual tasks behind an Application Load Balancer (ALB). Tasks scale horizontally based on CPU utilization and active WebSocket connection metrics.
* **Network Isolation**: The application runs inside a Private VPC. Databases (Amazon RDS PostgreSQL) and cache clusters (Amazon ElastiCache Redis) reside in isolated private subnets, accessible only from the ECS task security groups.
* **WebSocket Ingestion**: Managed via AWS API Gateway WebSocket API or an NGINX reverse proxy on ECS. The load balancer uses sticky sessions to maintain WebSocket connections with edge gateways and active dashboards.

---

## 3. Bandwidth Management Strategy

SME store owners typically operate on standard DSL or broadband lines with limited upload speeds (e.g. 5 Mbps). Standard high-definition video feeds consume ~4 Mbps *per camera*.

Our platform implements a **strict zero-stream-upload policy**:
1. **Local Frame Dropping**: The Edge Ingestion module decodes raw streams locally. Intermediate frames are dropped, capping analysis at 15 FPS.
2. **Crop-based Model Cascade**: Crops regions around tracked targets to reduce GPU inference cycles, running full-frame detection only on key intervals.
3. **No Raw Uploads**: No raw streams are uploaded to the WAN. Only text-based JSON metadata (alert timestamps, camera statuses, bounding box tracks) is sent.
4. **Edge-blurring & Heavy Compression**: Alert clips are restricted to a 45-frame sequence (3 seconds). Faces are blurred locally. The clip is compressed to H.264 MP4 with aggressive quantization parameter (QP) settings, ensuring files average **200KB - 400KB**.
5. **Bandwidth Footprint Calculation**:
   - 10 Alerts per hour = $10 \times 350\text{KB} = 3.5\text{MB}$ total upload per hour.
   - Continuous 8-camera RTSP upload = $8 \times 4\text{Mbps} \times 3600\text{s} \approx 14.4\text{GB}$ upload per hour.
   - **Result**: Bandwidth reduction of **99.97%**.

---

## 4. Over-The-Air (OTA) Updates & Configurations

Distributed edge nodes require seamless software and model updates without onsite technician visits.

* **Config Pull Sync**: The edge Behavior Controller maintains a persistent WebSocket connection to the Cloud Config Service. When a store manager updates detection zones on the React dashboard, the configuration changes are pushed instantly over the WebSocket as a lightweight JSON payload and written to the edge node's local SQLite database.
* **Docker Image Deployments**: Updates to the core software stack (e.g., ingestion fixes or behavior rule refinements) are packaged as new Docker images and pushed to a secure cloud registry (AWS ECR).
* **Edge Agent Daemon (Watchtower / Custom Updater)**: A lightweight agent runs on the edge OS outside the primary Docker Compose stack. It regularly checks the cloud registry for new image tags matching the current release channel. When an update is detected:
  1. The agent pulls the new Docker image in the background.
  2. During off-peak store hours (e.g., 2:00 AM local time), the agent restarts the Docker Compose stack using the new images.
  3. The local SQLite configuration cache and models directory are preserved using persistent Docker volumes.
  4. If the containers fail to start or report errors, the agent automatically rolls back to the previous stable image tag.
