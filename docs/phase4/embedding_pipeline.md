# Embedding Pipeline

## Model Selection Justification
We selected **DINOv2** (`dinov2_vits14`) over CLIP and SigLIP:
- **Vision-Only Self-Supervision**: DINOv2 is optimized for patch-level feature registration, which is critical for finding fine details on packaging (packaging variations, barcodes, logos) compared to CLIP, which prioritizes text-image alignment.
- **Speed & VRAM footprint**: `dinov2_vits14` generates 384-dimensional vectors quickly on standard GPUs.

## Transform Stages
1. **Crop**: Pad crop border bounds by 4px.
2. **Resize**: Interpolate to 224x224.
3. **Normalize**: Subtract ImageNet mean `[0.485, 0.456, 0.406]` and divide by std `[0.229, 0.224, 0.225]`.
4. **L2 Normalization**: Ensures that dot product directly corresponds to cosine similarity.
