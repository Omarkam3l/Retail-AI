FROM python:3.12-slim

# Install system dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configurations
COPY src/ ./src
COPY configs/ ./configs

# Expose metrics port
EXPOSE 8000

ENV PYTHONPATH=/app

# Run mock or camera pipeline driver (to be customized by deployment script)
CMD ["python", "-c", "from src.common.observability import start_metrics_server; start_metrics_server(8000); import time; time.sleep(3600)"]
