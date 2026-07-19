# Retail AI Surveillance Platform

An AI-powered video analytics platform that analyzes existing store CCTV streams to detect suspicious activities, minimize retail losses, and trigger real-time alerts without requiring proprietary smart camera hardware.

This platform is specifically designed as a cost-effective, modular, and customizable solution for Small and Medium Retailers (SMEs) to combat shoplifting and unauthorized intrusion.

## Core Capabilities
1. **Real-time Shoplifting & Concealment Detection**: Computer vision pipeline identifying suspicious behaviors (e.g. pocketing items, hiding products in backpacks).
2. **Loitering & Restricted Area Intrusion**: Alerting staff when individuals hover in high-value zones or enter staff-only rooms.
3. **Passive-CCTV Stream Processing**: Processing standard RTSP/RTMP camera feeds locally (edge processing) or via a cloud-based video ingestion engine.
4. **FastAPI Production backend**: Web server providing REST endpoints and real-time WebSockets for event streaming.
5. **Streamlit Live Dashboard**: Modern, dark-themed dashboard to monitor live camera streams, view metrics, track alerts, and adjust system settings.
6. **Robust Recording System**: Automatically captures JPEG snapshots and MP4 video clips with pre- and post-event windows around generated alerts.
7. **SQLite Storage Layer**: Thread-safe WAL-enabled local database with automated migrations tracking cameras, alerts, events, system logs, and benchmark runs.
8. **Product Recognition Engine**: Integrates DINOv2 visual features, cosine similarity, caching, and custom confidence metrics to classify bounding box crops into specific store products/SKUs.

---

## Repository Structure

```text
Retail-AI/
├── dashboard/              # Streamlit Web Dashboard
│   ├── pages/              # Home, Live Cameras, Alerts, Timeline, Metrics, Health, Benchmarks, Settings
│   ├── api_client.py       # Client for communicating with the backend API
│   └── styles.py           # Custom dark theme and UI component styles
│
├── demo/                   # Demo scripts and templates
│   ├── demo_rtsp.py        # RTSP stream processor
│   ├── demo_video.py       # Video file processor
│   ├── demo_webcam.py      # Live webcam stream processor
│   └── sample_config.yaml  # Complete configuration blueprint
│
├── docs/                   # Product & Engineering documentation
│   ├── phase1/             # System specifications and product requirements
│   ├── phase2/             # Design specs and prototyping
│   ├── phase3/             # Deployments & production plans
│   ├── phase4/             # Product Recognition details & pipeline models
│   │   ├── product_recognition.md
│   │   ├── embedding_pipeline.md
│   │   ├── similarity_search.md
│   │   ├── catalog_design.md
│   │   └── benchmark_results.md
│   ├── api_reference.md    # FastAPI REST/WebSocket API endpoints
│   ├── dashboard_guide.md  # Detailed Streamlit page functions and features
│   ├── deployment_guide.md # Local and production server setup instructions
│   ├── docker_guide.md     # Container setup and GPU configuration
│   ├── performance_guide.md# Model comparison, throughput scaling, and VRAM consumption
│   └── troubleshooting.md  # Resolutions for common edge failures
│
├── src/                    # Source code
│   ├── api/                # FastAPI application, routers, websocket manager, schemas
│   ├── alerts/             # Alert Policy Engine, face blurring, repositories, dispatcher
│   ├── association/        # Object Association Engine, tracking state matcher
│   ├── behavior/           # Behavior rule engine, loitering and concealment rules
│   ├── common/             # Logging manager, observability metrics, common base types
│   ├── config/             # Pydantic configuration validation
│   ├── database/           # SQLite connection, repository CRUD, migrations
│   ├── detection/          # YOLO11 inference wrapper
│   ├── evaluation/         # Benchmarking runner, metrics, confusion matrix, error analysis
│   ├── inference/          # Pipeline orchestrator, event bus, profiler
│   ├── ingestion/          # Video decoder, frame buffers, sampler
│   ├── monitoring/         # CPU/GPU resource collection monitor
│   ├── product_recognition/# Embedding catalog, DINOv2 model, similarity, matcher, caches
│   ├── recording/          # Video writer, snapshot/clip managers, retention policy
│   ├── risk/               # Risk collector, state machine, suppression, scoring
│   └── tracking/           # ByteTrack adapter and tracking manager
│
├── tests/                  # Pytest suite
│   ├── test_api.py         # REST endpoint and route validation
│   ├── test_database.py    # Database schema, CRUD, and migrations tests
│   ├── test_recording.py   # Clips, snapshots, and file retention tests
│   ├── test_websocket.py   # WebSocket event streaming manager tests
│   ├── test_monitoring.py  # System resource monitor testing
│   ├── test_catalog.py     # Product Catalog database operations
│   ├── test_embeddings.py  # Preprocessor and feature extraction tests
│   ├── test_similarity.py  # Cosine similarity vector matching
│   ├── test_matching.py    # Match ranking and threshold filtering
│   ├── test_unknown_detection.py # Unknown product logger
│   ├── test_cache.py       # Embedding LRU cache operations
│   ├── test_recognition_engine.py # Overall recognition flow coordinator
│   ├── test_pipeline_integration_product.py # Backward-compatible pipeline checks
│   └── ...                 # Component-specific unit and integration tests
│
├── Dockerfile              # Production multi-service Docker image builder
├── docker-compose.yml      # Orchestrates API, Dashboard, and persistent directories
├── install.py              # Automates workspace initialization and dependency checks
├── requirements.txt        # Full application python dependencies
└── yolo11s.pt              # YOLOv11 Model Weights
```

---

## Quick Start Setup

### Step 1: Install & Verify
Run the installation script to build the local directory structure, generate default configurations, download YOLO models, and verify hardware compatibility:
```bash
python install.py
```

### Step 2: Run with Docker (Recommended)
Launch the entire system (FastAPI API + Streamlit Dashboard) with a single command:
```bash
docker compose up --build
```
Access the services at:
- **Web Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 3: Run Locally (Development)
If running directly on the host machine:
```bash
# Start backend API
uvicorn src.api.application:app --host 0.0.0.0 --port 8000 --reload

# Start dashboard (in another terminal)
streamlit run dashboard/app.py --server.port 8501
```

---

## REST Endpoints Overview

For details, see the [API Reference Guide](docs/api_reference.md).

- `GET /health` — Service readiness check.
- `GET /system/status` — Live resource metrics (CPU, RAM, GPU, camera count).
- `GET /cameras` — Returns registered cameras and processing status.
- `POST /camera/register` — Registers a camera URL or webcam device.
- `POST /camera/start` — Initializes video ingestion for a camera.
- `POST /camera/stop` — Stops processing for a camera.
- `GET /alerts` — Paginated & filtered list of recorded alert incidents.
- `POST /infer/frame` — Directly runs the CV pipeline on a single frame.
- `POST /infer/video` — Submits a video file for batch offline processing.
- `GET /metrics` — Rolling stats for FPS, processing latency, and stage timings.
- `ws://localhost:8000/ws` — WebSocket server for streaming real-time alerts and metadata.
