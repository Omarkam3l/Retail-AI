# Deployment Guide

## Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- NVIDIA GPU with CUDA (optional, CPU fallback supported)
- FFmpeg

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python install.py
```

### 2. Start the API Server
```bash
uvicorn src.api.application:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```

### 4. Access
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| RETAIL_AI_API_KEY | retail-ai-dev-key-2024 | API authentication key |
| API_BASE_URL | http://localhost:8000 | API URL for dashboard |
| LOG_LEVEL | INFO | Logging level |

## Production Deployment
For production, use Docker Compose:
```bash
docker compose up -d
```
