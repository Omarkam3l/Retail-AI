# Codebase Folder Structure - Phase 2 Design Review

This document maps out the production-grade repository structure for the Retail AI Surveillance Platform, showing the organization of the edge pipeline, cloud backend, and configuration directories.

---

## 1. Directory Tree

```text
Retail-AI/
├── configs/
│   ├── edge_config.yaml         # Edge-side global parameters (FPS, model paths)
│   ├── logging_config.yaml      # Multi-module python logging parameters
│   └── zones_template.json      # Template for camera coordinate polygons
│
├── datasets/                    # Local training sets & validation clips (git-ignored)
│   ├── raw/
│   └── processed/
│
├── docs/                        # Project documentation
│   ├── phase1/                  # System requirements & market fit
│   └── phase2/                  # Technical design & architecture specs
│
├── models/                      # Fine-tuned YOLO weight files (git-ignored)
│   ├── yolo11n_detect.onnx      # Object detection TensorRT/ONNX export
│   └── yolo8n_pose.onnx         # Pose keypoints TensorRT/ONNX export
│
├── notebooks/                   # R&D Jupyter notebooks
│   ├── 01_eda_datasets.ipynb
│   └── 02_evaluate_heuristics.ipynb
│
├── src/                         # Core Source Code
│   ├── edge/                    # Local Node Applications
│   │   ├── __init__.py
│   │   ├── ingestion/           # Video stream capture & decoding
│   │   │   ├── __init__.py
│   │   │   ├── decoder.py
│   │   │   └── rtsp_client.py
│   │   │
│   │   ├── pipeline/            # AI model execution
│   │   │   ├── __init__.py
│   │   │   ├── detector.py      # YOLO Wrapper (TensorRT)
│   │   │   ├── tracker.py       # ByteTrack Wrapper
│   │   │   └── pose_estimator.py# YOLO-Pose Wrapper
│   │   │
│   │   ├── behavior/            # Logic & State Machine
│   │   │   ├── __init__.py
│   │   │   ├── rules_engine.py  # Spatial relations & heuristics
│   │   │   └── state_machine.py # Tracking lifecycle states
│   │   │
│   │   └── controller/          # Main Edge orchestrator daemon
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── local_db.py      # SQLite CRUD operations
│   │       └── sync_agent.py    # Cloud API WebSocket / HTTPS sync
│   │
│   └── cloud/                   # Backend Microservices (FastAPI)
│       ├── __init__.py
│       ├── common/              # Database models, configuration, security
│       │   ├── db.py            # PostgreSQL connection pool
│       │   ├── security.py      # JWT validation
│       │   └── models.py        # SQLAlchemy schema models
│       │
│       ├── alert_service/       # Alert routing & ingestion
│       │   ├── main.py
│       │   └── s3_uploader.py   # S3 storage integrations
│       │
│       └── config_service/      # Edge configuration synchronizations
│           └── websocket.py     # Connection manager
│
├── tests/                       # Automated Test Suite
│   ├── edge/
│   │   ├── test_decoder.py
│   │   ├── test_rules.py
│   │   └── test_state_machine.py
│   │
│   └── cloud/
│       ├── test_auth.py
│       └── test_alert_api.py
│
├── Dockerfile.edge              # Multi-stage edge build
├── Dockerfile.cloud             # Multi-stage cloud build
├── docker-compose.yml           # Edge services deployment configuration
├── requirements.txt             # Project packages
└── README.md                    # Setup & Quickstart Guide
```

---

## 2. Component Integration Description

1. **RTSP to Frames**: `edge/ingestion` contains the background threads reading live IP feeds and pushing raw numpy frames to the shared memory queue defined in `decoder.py`.
2. **Frames to Metadata**: `edge/pipeline` pulls frames, runs `detector.py`, passes bounding boxes to `tracker.py`, crops persons, runs `pose_estimator.py`, and outputs a consolidated frame object metadata dictionary.
3. **Metadata to State Events**: `edge/behavior/rules_engine.py` ingests the frames' metadata, evaluates wrist distances and overlapping shapes, and updates the customer lifecycle trackers inside `state_machine.py`.
4. **Events to Cloud Routing**: When an alert fires, `edge/controller/sync_agent.py` applies face blurring, encodes the video, and posts the payload to the Cloud `alert_service`.
5. **Orchestrator Control**: `edge/controller/main.py` is the bootstrap daemon initializing the frame queue, spawning the ingestion and inference threads, and running the pipeline.
