# API Reference

Base URL: `http://localhost:8000`

## Authentication
Include `X-API-Key` header with all requests (except /health).

## Endpoints

### Health
- `GET /health` — Service health check
- `GET /system/status` — Detailed system status (CPU, GPU, memory)

### Cameras
- `GET /cameras` — List all registered cameras
- `POST /camera/register` — Register a new camera
- `POST /camera/start?camera_id=X` — Start camera processing
- `POST /camera/stop?camera_id=X` — Stop camera processing

### Alerts
- `GET /alerts?page=1&page_size=50&level=HIGH` — List alerts with filtering

### Inference
- `POST /infer/frame` — Run inference on a single image (multipart upload)
- `POST /infer/video` — Upload video for batch processing

### Metrics
- `GET /metrics` — Pipeline performance metrics

### WebSocket
- `ws://localhost:8000/ws` — Real-time event stream
