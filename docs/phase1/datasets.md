# Datasets and References - Retail AI Surveillance Platform

This document catalogs public computer vision datasets suitable for pre-training, fine-tuning, and evaluating the Retail AI Surveillance Platform.

---

## 1. Object Detection Datasets

### 1.1 COCO (Common Objects in Context)
* **Purpose**: General object detection, instance segmentation, and keypoint localization. Contains classes relevant to retail (e.g., `person`, `backpack`, `handbag`, `umbrella`, `hand`, `bottle`).
* **Size**: 330,000+ images (200,000+ labeled), 1.5 million object instances across 80 categories.
* **License**: Creative Commons Attribution 4.0.
* **Annotations**: 2D bounding boxes, polygon segmentation masks, and 17-point human keypoints.
* **Advantages**:
  - Gold standard benchmark with high image diversity.
  - Pre-trained weights (YOLO, Faster R-CNN) are widely available.
* **Disadvantages**:
  - Lacks specific retail environments (e.g., crowded store aisles, overhead camera angles).
* **Recommended Usage**: Used for pre-training the base object detector model (YOLOv8/v10/v11) to detect humans and basic carrying gear (backpacks, purses).

---

## 2. Multi-Object Tracking Datasets

### 2.1 MOT20 (Multi-Object Tracking Challenge)
* **Purpose**: Tracking multiple pedestrians in extremely dense crowd settings.
* **Size**: 8 video sequences (4 training, 4 testing) containing over 13,000 frames.
* **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
* **Annotations**: 2D bounding boxes, tracking ID trajectories, visibility ratios.
* **Advantages**:
  - Trains models to handle extreme crowd density, occlusions, and person-to-person intersections.
* **Disadvantages**:
  - Filmed primarily in public plazas or outdoor markets; camera heights differ from standard indoor dome CCTV cameras.
* **Recommended Usage**: Fine-tuning the tracking association model (e.g., ByteTrack, DeepSORT re-identification layers) to maintain customer tracking IDs when they pass behind pillars or other shoppers.

---

## 3. Pose Estimation Datasets

### 3.1 MPII Human Pose Dataset
* **Purpose**: 2D human pose estimation of joint coordinates.
* **Size**: ~25,000 images containing over 40,000 people with annotated joints.
* **License**: Personal and Non-Commercial Research use.
* **Annotations**: 2D joint coordinates (16 body joints including head, neck, shoulders, elbows, wrists, hips, knees, ankles).
* **Advantages**:
  - Covers a wide variety of human activities and body shapes.
  - Highly robust keypoint coordinates.
* **Disadvantages**:
  - Static images only. Cannot capture temporal movement of hands reaching into bags.
* **Recommended Usage**: Fine-tuning the skeleton pose estimator to track wrist-to-pocket and wrist-to-hip spatial distances.

---

## 4. Action Recognition Datasets

### 4.1 UCF101
* **Purpose**: General video action recognition.
* **Size**: 13,320 videos across 101 action categories.
* **License**: Open research license (non-commercial).
* **Annotations**: Video-level action labels (no bounding boxes).
* **Advantages**:
  - High action variety and widely supported by video classification networks (C3D, I3D).
* **Disadvantages**:
  - Video clips are low resolution, and actions are non-retail (e.g., sports, playing instruments).
* **Recommended Usage**: Training base temporal feature extraction weights in spatio-temporal video networks.

---

## 5. Retail Surveillance & Shopping Action Datasets

### 5.1 MERL Shopping Dataset
* **Purpose**: Fine-grained shopping action recognition from overhead retail cameras.
* **Size**: 96 video sequences (each ~2 minutes, 1080p, 30 FPS) showing customers interacting with items on shelves.
* **License**: MERL Research License.
* **Annotations**: Frame-level annotations for five actions: `Reach to shelf`, `Retrieve product from shelf`, `Inspect product`, `Return product to shelf`, and `Put product into cart/basket`.
* **Advantages**:
  - Filmed in an actual supermarket environment from top-down angles mimicking real security cameras.
  - Frame-accurate starting and ending times for reaching gestures.
* **Disadvantages**:
  - Limited number of actors and single shelf setting. Does not contain explicit "pocket concealment" actions.
* **Recommended Usage**: Core training dataset for the reaching-and-grabbing gesture classification head.

### 5.2 DAiS (Darmstadt Action in Supermarket)
* **Purpose**: Tracking supermarket shoppers and detecting checkout/shopping activities.
* **Size**: 15+ hours of multi-camera recordings of grocery store lanes.
* **License**: Open Research License.
* **Annotations**: 2D bounding boxes for products, shopping carts, and shoppers; action event stamps.
* **Advantages**:
  - Multi-camera coverage allows testing of cross-camera tracking algorithms.
* **Disadvantages**:
  - Complicated data structure and variable lighting quality.
* **Recommended Usage**: Validating multi-camera shopper association models and cart interaction rules.

### 5.3 RetailAction Dataset
* **Purpose**: Detecting theft-related behaviors in retail stores.
* **Size**: ~2,500 annotated video clips of simulated shoplifting (pocketing, backpack stuffing, product swapping) and normal shopping behaviors.
* **License**: Open Research/Academic License.
* **Annotations**: Person bounding boxes, item bounding boxes, and action sequence labels (e.g., `concealment_start`, `concealment_end`).
* **Advantages**:
  - Contains explicit negative and positive theft behaviors.
* **Disadvantages**:
  - Simulated actions (staged by actors), which may differ slightly in speed and hesitation from genuine shoplifting events.
* **Recommended Usage**: Primary training dataset for the concealment behavior classifier.
