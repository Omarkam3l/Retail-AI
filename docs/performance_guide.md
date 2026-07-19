# Performance Guide

## Optimization Tips

### GPU Acceleration
- Set `device: cuda` in configuration
- Enable FP16: `fp16: true`
- Use TensorRT for maximum throughput

### Model Selection
| Model | Speed | Accuracy | Use Case |
|---|---|---|---|
| yolo11s | ⚡ Fast | Good | Real-time (recommended) |
| yolo11m | 🔄 Medium | Better | Balanced |
| yolo11l | 🐢 Slow | Best | Offline analysis |

### Memory Management
- Set retention policy to auto-clean old clips
- Use WAL mode for SQLite (enabled by default)
- Monitor with System Health dashboard page

### Multi-Camera Scaling
- Each camera runs in its own thread
- GPU memory scales linearly with camera count
- Recommend max 4-8 cameras per GPU
