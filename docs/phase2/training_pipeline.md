# Model Training & Quantization Pipeline - Phase 2 Design Review

This document specifies the pipeline for data preprocessing, data augmentation, transfer learning, fine-tuning schedules, and edge optimization via model quantization.

---

## 1. Core Training Flow

To achieve high accuracy on custom retail objects (backpacks, carrying gear, specific retail products) and behaviors (item concealment), we leverage transfer learning from pre-trained COCO weights, followed by fine-tuning on custom labeled retail surveillance datasets.

```text
┌─────────────────┐
│  Pre-trained    │ (COCO Weights)
│  YOLOv11 Models │
└────────┬────────┘
         │
         ▼
 ┌───────────────┐     ┌───────────────────────┐
 │ Custom Retail ├────>│ Synthetic Data        │ (Geometric, blur, occlusion)
 │ Datasets      │     │ Augmentations         │
 └───────────────┘     └──────────┬────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
                       │ Transfer Learning     │ (Freeze backbones first,
                       │ & Fine-Tuning         │  then unfreeze & train)
                       └──────────┬────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
                       │ Model Quantization    │ (Post-Training Quantization
                       │ & Optimization        │  FP16/INT8 via TensorRT)
                       └──────────┬────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
                       │ Edge Node Deployment  │
                       └───────────────────────┘
```

---

## 2. Data Augmentation Strategies

Retail surveillance cameras are fixed, high-angle sensors subject to variable lighting (fluorescent store lamps, direct sunlight from windows) and frequent crowd occlusions. We apply offline and online augmentations to improve generalization:

* **Geometric Transformations**: Random cropping (simulating varying camera distances), rotation (accounting for variable camera mounting angles $\pm15^\circ$), and horizontal flipping.
* **Lighting & Color Jittering**: Random brightness adjustments ($\pm30\%$), contrast variations, and Hue-Saturation-Value (HSV) adjustments to simulate different store lighting.
* **Camera Artifacts**: Gaussian noise, motion blur (simulating rapid hand movement during concealment), and compression artifacts (simulating lower-quality analog CCTV feeds).
* **Synthetic Occlusions (Mosaic & MixUp)**: Combining four training images into one or overlaying semi-transparent item bboxes on person bodies to force the detector to find objects that are partially hidden by limbs or shopping carts.

---

## 3. Transfer Learning & Fine-Tuning Schedule

### 3.1 Object Detection (YOLOv11n)
* **Initial Stage (Epochs 0 - 30)**: Freeze the backbone layers (features extractor). Train only the neck and detection head using a learning rate of $10^{-3}$ and AdamW optimizer. This preserves basic visual feature weights (lines, shapes) while adjusting the classifier to target classes.
* **Fine-Tuning Stage (Epochs 30 - 150)**: Unfreeze all layers. Train the entire network using a cosine annealing learning rate scheduler dropping from $10^{-4}$ down to $10^{-6}$.
* **Loss Functions**: 
  - Complete Intersection over Union (CIoU) loss for bounding box regression.
  - Distribution Focal Loss (DFL) to handle bounding box uncertainty.
  - Binary Cross-Entropy (BCE) loss for class classification.

### 3.2 Pose Estimation (YOLOv8-Pose / YOLOv11-Pose)
* Fine-tune the pre-trained pose model specifically on retail-labeled skeletons (arms reaching, bending at waist) using the Object Keypoint Similarity (OKS) loss function, which weights keypoint errors according to the visibility and standard deviation of each joint type (e.g. wrists have higher weight variance than ears).

---

## 4. Model Quantization Pipeline (Edge Optimization)

Raw deep learning models run in standard 32-bit floating point precision (FP32). To execute inference within our edge latency budget (<65ms) on a Jetson Orin Nano or Intel NUC CPU, the models must be quantized to 16-bit (FP16) or 8-bit integer (INT8) representation.

### 4.1 Export to ONNX
Compile the PyTorch weights (`.pt`) to Open Neural Network Exchange (ONNX) format:
```bash
yolo export model=yolo11n.pt format=onnx half=True dynamic=True
```

### 4.2 TensorRT Compilation (Jetson Edge Nodes)
For edge deployment on NVIDIA Jetson, we compile the ONNX graphs to highly optimized TensorRT engine engines:
* **FP16 Quantization**:
  ```bash
  trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n_fp16.engine --fp16
  ```
  - *Pros*: Near-zero drop in mAP accuracy; massive speedup (roughly 2.5x faster than FP32).
* **INT8 Quantization (Post-Training Quantization - PTQ)**:
  Requires a calibration dataset of at least 1,000 representative store frames. The calibration process determines the dynamic range of each activation layer to minimize quantization noise when converting weights from float to integer.
  ```bash
  trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n_int8.engine --int8 --calib=calib_dataset.cache
  ```
  - *Pros*: Maximum inference speed (up to 4.5x faster than FP32), minimal memory footprints.
  - *Cons*: Slight accuracy drop (typically 0.5% to 1.5% mAP degradation).
* **MVP Decision**: For the initial release, we deploy models using **FP16 quantization**, which yields sufficient speed (6ms per frame YOLO detection) on the Jetson Orin Nano without requiring complex INT8 calibration pipelines.
