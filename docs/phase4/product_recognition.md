# Product Recognition Engine

The Product Recognition Engine extends the platform by classifying bounding box crops into specific SKUs, brand catalogs, and packaging types rather than generic class labels.

## Architecture Flow
1. **Detection**: Bounding box proposals (e.g. SHELF_ITEM, PERSON).
2. **Crop**: Extract bounding box region with border padding.
3. **Preprocessing**: Scale and normalize for the embedding neural net.
4. **Feature Extraction**: DINOv2 transforms visual details to a 384-dimensional vector.
5. **Similarity Search**: Cosine distance comparison against the Product Catalog index.
6. **Confidence Fusion**: Considers crop quality, size, and YOLO detection metrics.
