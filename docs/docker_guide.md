# Docker Guide

## Build and Run
```bash
# Build and start all services
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## Services
| Service | Port | Description |
|---|---|---|
| api | 8000 | FastAPI backend |
| dashboard | 8501 | Streamlit dashboard |

## Volumes
- `./configs` → `/app/configs` — Configuration files
- `./data` → `/app/data` — Database, clips, snapshots
- `./logs` → `/app/logs` — Application logs

## GPU Support
For NVIDIA GPU support, update docker-compose.yml:
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
