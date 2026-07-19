# Troubleshooting Guide

## Common Issues

### API won't start
- Check port 8000 is available: `netstat -an | findstr 8000`
- Verify Python path: `echo $PYTHONPATH`
- Check logs: `logs/api.log`

### Dashboard can't connect to API
- Ensure API is running first
- Check API_BASE_URL environment variable
- In Docker: dashboard connects to `http://api:8000`

### No GPU detected
- Install NVIDIA drivers
- Install CUDA toolkit
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Video processing slow
- Use GPU: set device to "cuda"
- Lower confidence threshold to reduce post-processing
- Use a smaller model (yolo11s vs yolo11l)

### Out of memory
- Reduce batch size
- Use FP16 inference
- Lower input resolution

### Docker build fails
- Ensure Docker daemon is running
- Check disk space: `docker system df`
- Clear cache: `docker builder prune`
