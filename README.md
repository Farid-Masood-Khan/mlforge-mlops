# MLForge — End-to-End MLOps Pipeline & Production System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![Tests: Pytest](https://img.shields.io/badge/tests-26%2F26%20passing-brightgreen.svg)](https://docs.pytest.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11.0-0194E2.svg)](https://mlflow.org/)

A production-grade machine learning system that takes a tabular customer churn classification model all the way from raw data ingestion to a versioned, monitored, containerised API — with every single run reproducible end to end.

> **Core Engineering Principle:** Notebooks prove you can fit a model. MLForge proves you can *ship* one — validate data at entry, track experiment lineage, register models, serve via a typed API, enforce CI gates, and monitor for distribution drift in production.  
> 
> 💡 **Domain Baseline**: This MLOps pipeline represents the continuous delivery evolution of our supervised baseline project, **[Customer-Churn-Prediction](https://github.com/Farid-Masood-Khan/Customer-Churn-Prediction)**.

---

## 1. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Raw Data Drop  │ ──▶ │ Validation Gate  │ ──▶ │ Unified Pipeline     │
│  (Telco Churn)  │     │ (Schema/Ranges)  │     │ (Preprocess + Model) │
└─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ FastAPI Serving │ ◀── │ MLflow Registry  │ ◀── │ Experiment Tracking  │
│ (/predict, /dr) │     │ (Champion/Chall) │     │ (Params, ROC, Brier) │
└────────┬────────┘     └──────────────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────┐
│ Drift Monitor   │ ──▶ Triggers Alert / Retrain on PSI ≥ 0.25
│ (KS-test & PSI) │
└─────────────────┘
```

---

## 2. Key Features & Capabilities

- **Data Validation Gate (`src/data/validate.py`)**: Incoming datasets are validated against strict declared schema constraints (ranges, non-null requirements, column types, categorical domains). Dirty data fails loudly at the boundary.
- **Unified Feature Pipeline (`src/features/build.py`)**: A single `scikit-learn` `Pipeline` (imputation $\rightarrow$ scaling/encoding $\rightarrow$ model) fitted once and serialised as one immutable artifact, eliminating training/serving skew by construction.
- **Experiment Tracking & Model Registry (`src/models/`)**: Every training run logs parameters, metrics (ROC-AUC, PR-AUC, Brier score), artifacts, and code hashes to MLflow. Models are promoted via `champion`/`challenger` aliases.
- **Drift Detection Engine (`src/monitoring/drift.py`)**: Computes two-sample Kolmogorov-Smirnov (KS) statistics for continuous features and Population Stability Index (PSI) for binned distributions against reference baselines. Automatically alerts when $\text{PSI} \ge 0.25$.
- **Production FastAPI Server (`src/serving/app.py`)**: High-throughput REST API with Pydantic v2 schemas exposing `/predict`, `/predict/batch`, `/health`, `/model/info`, and `/metrics`.
- **Streamlit Triage Dashboard (`demo/app.py`)**: Interactive UI for simulating live customer inputs, testing latency, and reviewing drift reports.

---

## 3. Measured Empirical Results

All metrics are logged to `results/metrics_summary.json` and reproducible via `src/reproduce.py`:

| Model Variant | Test ROC-AUC | Test PR-AUC | Brier Score | Status |
|---|---|---|---|---|
| **HistGradientBoosting (Champion)** | **0.8462** | **0.6581** | **0.1348** | **Promoted to Production** |
| Logistic Regression (Baseline) | 0.8415 | 0.6504 | 0.1372 | Candidate |
| Naive Class Baseline | 0.5000 | 0.2654 | 0.1950 | Rejected |

- **Deterministic Reproduction Test**: Evaluated across 1,409 holdout rows. Maximum prediction probability delta between original and re-trained model: **0.000000** (bit-exact reproducibility).

---

## 4. Quickstart & Verification

```powershell
# 1. Install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Run unit & integration test suite (26 tests)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "test"

# 3. Static type check (Mypy)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "typecheck"

# 4. Code quality & lint (Ruff)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "lint"

# 5. Launch FastAPI prediction server
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "app"
```

---

## 5. Repository Structure

```
mlforge-mlops/
├── README.md                  # System overview and quickstart
├── pyproject.toml             # Ruff, Mypy, and Pytest configuration
├── requirements.txt           # Pinned dependencies
├── run.ps1                    # Standardized PowerShell runner
├── Dockerfile.api             # Containerized FastAPI runtime
├── Dockerfile.train           # Containerized training job
├── docker-compose.yml         # API + MLflow orchestration
├── configs/
│   └── train_config.yaml      # Hyperparameters and threshold settings
├── data/
│   ├── raw/                   # Data provenance cards
│   └── reference/             # Baseline reference snapshot for drift
├── src/
│   ├── data/                  # Download and validation gates
│   ├── features/              # Preprocessing pipelines
│   ├── models/                # Training, evaluation, and registry promotion
│   ├── monitoring/            # KS-test and PSI drift detection
│   └── serving/               # FastAPI application and predictor wrappers
├── tests/                     # 26 automated unit and API integration tests
└── results/                   # Reproducible metrics artifacts
```
