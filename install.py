"""System installer: downloads models, creates folders, generates configs, verifies dependencies."""
import os
import sys
import subprocess
import shutil
import json


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def create_directories() -> None:
    """Creates required directory structure."""
    dirs = [
        "data", "data/clips", "data/snapshots", "data/uploads",
        "logs", "configs", "models"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅ Created: {d}/")


def generate_default_config() -> None:
    """Generates default YAML configuration file."""
    config = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "api_key": "retail-ai-dev-key-2024",
            "cors_origins": ["*"]
        },
        "detection": {
            "model_path": "yolo11s.pt",
            "confidence_threshold": 0.35,
            "device": "auto",
            "fp16": True
        },
        "tracking": {
            "track_threshold": 0.25,
            "max_age": 30
        },
        "behavior": {
            "loiter_threshold_seconds": 20.0,
            "concealment_gap_seconds": 10.0
        },
        "risk": {
            "medium_threshold": 0.5,
            "high_threshold": 0.7
        },
        "recording": {
            "pre_event_frames": 90,
            "post_event_frames": 60,
            "retention_hours": 72,
            "max_storage_gb": 10
        },
        "dashboard": {
            "port": 8501,
            "theme": "dark"
        }
    }

    import yaml
    config_path = os.path.join("configs", "default.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✅ Generated: {config_path}")


def download_model() -> None:
    """Downloads YOLO11s model weights if not present."""
    model_path = "yolo11s.pt"
    if os.path.exists(model_path):
        print(f"  ✅ Model already exists: {model_path}")
        return

    print("  ⬇️  Downloading YOLO11s model...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11s.pt")
        print(f"  ✅ Model downloaded: {model_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to download model: {e}")
        print("     You can manually download from: https://github.com/ultralytics/assets")


def verify_cuda() -> None:
    """Checks CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available: {gpu_name}")
        else:
            print("  ⚠️  CUDA not available — will use CPU")
    except ImportError:
        print("  ❌ PyTorch not installed")


def verify_ffmpeg() -> None:
    """Checks FFmpeg installation."""
    if shutil.which("ffmpeg"):
        print("  ✅ FFmpeg found")
    else:
        print("  ⚠️  FFmpeg not found — video recording may not work")


def verify_opencv() -> None:
    """Checks OpenCV installation."""
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ❌ OpenCV not installed")


def verify_torch() -> None:
    """Checks PyTorch installation."""
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not installed")


def main():
    print_header("Retail AI Surveillance Platform — Installer")

    print("[1/6] Creating directories...")
    create_directories()

    print("\n[2/6] Generating default configuration...")
    generate_default_config()

    print("\n[3/6] Downloading model weights...")
    download_model()

    print("\n[4/6] Verifying CUDA...")
    verify_cuda()

    print("\n[5/6] Verifying dependencies...")
    verify_ffmpeg()
    verify_opencv()
    verify_torch()

    print("\n[6/6] Installation complete!")
    print_header("Quick Start")
    print("  1. Start API:       uvicorn src.api.application:app --reload")
    print("  2. Start Dashboard: streamlit run dashboard/app.py")
    print("  3. Docker:          docker compose up")
    print("  4. Demo:            python scripts/run_demo.py --video data/sample.mp4")


if __name__ == "__main__":
    main()
