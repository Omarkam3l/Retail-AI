FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, dashboard, configs, and scripts
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY demo/ ./demo/

# Create data directories
RUN mkdir -p data/clips data/snapshots data/uploads logs

# Expose ports: API (8000) + Dashboard (8501)
EXPOSE 8000 8501

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
