# Retail AI Surveillance Platform

An AI-powered video analytics platform that analyzes existing store CCTV streams to detect suspicious activities, minimize retail losses, and trigger real-time alerts without requiring proprietary smart camera hardware.

This platform is specifically designed as a cost-effective, modular, and customizable solution for Small and Medium Retailers (SMEs) to combat shoplifting and unauthorized intrusion.

## Core Capabilities
1. **Real-time Shoplifting & Concealment Detection**: Computer vision pipeline identifying suspicious behaviors (e.g. pocketing items, hiding products in backpacks).
2. **Loitering & Restricted Area Intrusion**: Alerting staff when individuals hover in high-value zones or enter staff-only rooms.
3. **Passive-CCTV Stream Processing**: Processing standard RTSP/RTMP camera feeds locally (edge processing) or via a cloud-based video ingestion engine.

## Repository Structure

```text
Retail-AI/
│
├── docs/
│   ├── phase1/             # System specifications & product documentation
│   │   ├── problem_statement.md
│   │   ├── market_research.md
│   │   ├── competitors.md
│   │   ├── use_cases.md
│   │   ├── requirements.md
│   │   ├── datasets.md
│   │   ├── architecture_overview.md
│   │   └── success_metrics.md
│   │
│   ├── phase2/             # Design specs & prototyping
│   ├── phase3/             # Deployments & production plans
│   └── images/             # Architecture diagrams & mockups
│
├── src/                    # Source code (ingestion, detection, alerts)
├── datasets/               # Local training/fine-tuning datasets (git-ignored)
├── models/                 # Fine-tuned YOLO weight files (git-ignored)
├── notebooks/              # Jupyter notebooks for model prototyping & evaluation
├── tests/                  # Pytest unit and integration test suite
├── configs/                # Camera RTSP URLs & alert threshold configs
├── requirements.txt        # Python package dependencies
├── README.md               # Home documentation page
└── .gitignore              # Git ignore patterns
```

## Setup & Running
See [docs/phase1/requirements.md](docs/phase1/requirements.md) for deployment prerequisites and [docs/phase1/architecture_overview.md](docs/phase1/architecture_overview.md) for data flow specifications.
