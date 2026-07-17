# Project Scope Specification - Phase 3 Production Planning

**Document Type**: Scope Boundary Document  
**Classification**: Product Planning Reference  
**Audience**: Product Managers, Developers, Stakeholders  

This document defines the functional boundaries of the Retail AI Surveillance Platform, distinguishing the MVP requirements from future iterations to ensure timely deployment and execution.

---

## 1. Scope Matrix

```text
┌──────────────────────────────────────┐
│                MVP                   │
│ • Single-camera tracking             │
│ • Concealment & Intrusion detection  │
│ • Rule-based behavior heuristics     │
│ • Edge Docker + Cloud API & Dashboard│
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             Post-MVP / v2.0          │
│ • Cross-camera handovers             │
│ • ST-GCN learned gesture models      │
│ • POS transaction cross-referencing  │
│ • Native Mobile iOS/Android apps     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│            Future / v3.0             │
│ • Graph Neural Networks (Scene GNN)  │
│ • Generative AI incident reports     │
│ • 3D pose and depth camera support   │
└──────────────────────────────────────┘
```

---

## 2. MVP Scope (Phase 1)
* **Single-Camera Tracking**: Track shoppers and items within the field of view of a single camera. No multi-camera stitching.
* **Basic Gesture Concealment**: Detect item pocketing and personal bag stuffing using 2D spatial rules and hand-pose proximity.
* **Zone Intrusion & Loitering**: Polygon-based restricted area and loitering warnings.
* **Edge-Cloud Hybrid Architecture**: Local decoding/models, cloud database, WebSocket alert delivery, and face-blurred looping video clips.
* **Web Admin Dashboard**: React-based portal to view alerts, draw zones, and manage streams.

---

## 3. Post-MVP & Version 2.0 Scope
* **Cross-Camera Shopper Association (Re-ID)**: Track a customer ID as they walk between different cameras (handover tracking).
* **ST-GCN Action Models**: Replace hand-coded behavior rules with trained Spatio-Temporal Graph Convolutional Networks.
* **Point-of-Sale (POS) Integration**: Cross-reference product pickup events with register transactions in real-time to suppress false alerts for paid items.
* **Native Mobile Applications**: Dedicated Swift/Kotlin apps with low-level background push notification waking systems for store associates.

---

## 4. Version 3.0 & Future Scope
* **Scene Graph Neural Networks (Scene GNN)**: Fully model aisles, shoppers, and product relations dynamically.
* **LLM Incident Summaries**: Autocomplete narrative reports for police/insurance detailing the sequence of actions.
* **3D Pose & Depth Camera Support**: Use depth sensors to eliminate 2D projection overlap errors.

---

## 5. Justification for Excluded Features

### 5.1 Why Multi-Camera Tracking (Re-ID) is Excluded from MVP
* *Complexity*: Cross-camera re-identification requires computing deep appearance embeddings for every person and comparing them globally in real-time. This increases edge VRAM/CPU usage by $\geq 300\%$ and introduces high latency.
* *Justification*: Most shoplifting events occur within a single aisle/camera zone (cosmetics, alcohol sections). Single-camera tracking is sufficient to detect the gesture. Adding cross-camera tracking is an optimization, not a blocker for the core value.

### 5.2 Why POS Integration is Excluded from MVP
* *Complexity*: Retailers operate dozens of proprietary POS systems (NCR, Toshiba, Shopify, Clover). Integrating with them requires custom APIs, API coordination layer, and database sync pipelines for each store.
* *Justification*: SMEs need a plug-and-play solution that does not require IT integration with their financial registers. Suppressing alerts via manual associate confirmation is a robust, zero-integration alternative for MVP.

### 5.3 Why ST-GCN Action Models are Excluded from MVP
* *Complexity*: Graph GCN models require large volumes of labeled frame-sequence skeleton data to train. Tuning their thresholds to generalize across different body shapes and camera heights is a long research process.
* *Justification*: Rule-based geometric heuristics are 100% explainable, deterministic, compile instantly, and can be easily adjusted by developers. They provide a predictable baseline for the MVP, which can then be used to collect training samples for future ST-GCN models.
