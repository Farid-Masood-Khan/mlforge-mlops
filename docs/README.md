# MLForge — End-to-End MLOps Pipeline

A production-style machine learning system that takes a tabular classification model (customer churn by default, swappable for fraud) all the way from raw data to a versioned, monitored, containerised API — with every run reproducible end to end.

> **The point of this project:** Notebooks prove you can fit a model. MLForge proves you can *ship* one — validate the data, track every experiment, register the model, serve it behind an API, test it in CI, and know when it stops working.

## Why production ML matters

Most portfolio ML projects die at `model.fit()`. Real ML work is the other 90%: data contracts, experiment lineage, deployment packaging, and detecting when the world changes under a trained model. MLForge implements that 90% at small scale, with the same tools used in industry (MLflow, FastAPI, Docker, GitHub Actions), so a reviewer can read the repo and see a system, not a script.

## Features

- **Data validation gate** — incoming datasets are checked against a declared schema (column names, dtypes, ranges, missing-value ceilings, target leakage checks) before any training happens. Dirty data fails loudly, never silently.
- **Feature pipeline** — a single `sklearn` `Pipeline` (imputation → scaling/encoding → model) that is fitted once and shipped as one artefact, so training/serving skew is impossible by construction.
- **Experiment tracking & model registry** — every training run is logged to MLflow (params, metrics, artefact, data hash, code hash, seed). Promoted models are registered with aliases (`champion` / `challenger`) instead of hard-coded paths.
- **FastAPI serving** — `/predict`, `/predict/batch`, `/health`, `/model/info`, `/metrics` endpoints with request validation (Pydantic), structured logging, and the model loaded from the MLflow registry at startup.
- **Docker packaging** — one multi-stage `Dockerfile` for the API, one for the training job, `docker-compose.yml` to run API + MLflow tracking server locally in two commands.
- **CI on every push** — GitHub Actions: lint (ruff), type-check (mypy on `src/`), unit + API tests (pytest), data-validation smoke test, and a Docker build. A manual `retrain` workflow runs the full pipeline on a fresh data drop.
- **Drift & performance monitoring** — a scheduled job compares live prediction inputs and score distributions against the training reference (PSI per feature, prediction-rate shift). Breaching thresholds writes a drift report and (in the stretch scope) opens a GitHub issue / triggers retraining.
- **Reproducibility as a feature** — pinned dependencies (`requirements.txt` with hashes or `uv.lock`), fixed global seed, dataset content-hash recorded with every run, `make reproduce` re-runs the exact champion pipeline from the recorded data hash.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Ecosystem standard for ML |
| Modelling | scikit-learn (LogReg baseline → HistGradientBoosting / XGBoost) | Strong tabular baselines, clean `Pipeline` API |
| Data validation | pandera (or pydantic-based custom checks) | Declarative schemas, readable failures |
| Tracking / registry | MLflow (local SQLite + local artefact store; Dockerised) | Industry default, free, self-contained |
| Serving | FastAPI + uvicorn | Async, typed, auto OpenAPI docs |
| Packaging | Docker + docker-compose | Identical runtime everywhere |
| CI | GitHub Actions | Free for public repos, where the reviewers are |
| Testing | pytest, httpx (TestClient), ruff, mypy | Standard, fast |
| Demo / monitoring UI | Streamlit (thin client over the API) | Reviewer-friendly without frontend overhead |
| Default dataset | IBM Telco Customer Churn (public, ~7k rows) | Well-known, so reviewers can sanity-check results |

## Planned repo structure

```
mlforge/
├── README.md                  ← you are here
├── Makefile                   # setup / train / serve / test / reproduce
├── docker-compose.yml         # api + mlflow services
├── Dockerfile.api
├── Dockerfile.train
├── requirements.txt           # pinned
├── pyproject.toml             # ruff / mypy / pytest config
├── configs/
│   └── train_config.yaml      # model params, thresholds, paths (no magic numbers in code)
├── data/
│   ├── raw/                   # git-ignored; data card + download script instead
│   └── reference/             # reference snapshot used for drift baselines (small, versioned)
├── src/
│   ├── data/
│   │   ├── download.py        # fetches dataset, writes data card with sha256
│   │   └── validate.py        # schema checks (pandera)
│   ├── features/
│   │   └── build.py           # returns the fitted preprocessing Pipeline
│   ├── models/
│   │   ├── train.py           # trains, logs to MLflow, registers model
│   │   ├── evaluate.py        # holdout metrics, threshold selection
│   │   └── promote.py         # champion/challenger promotion rules
│   ├── serving/
│   │   ├── app.py             # FastAPI app
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── predictor.py       # registry loading + inference wrapper
│   └── monitoring/
│       ├── drift.py           # PSI + distribution comparisons
│       └── report.py          # writes drift report (JSON + markdown summary)
├── tests/
│   ├── test_validate.py
│   ├── test_features.py
│   ├── test_train_smoke.py    # tiny-data end-to-end train
│   └── test_api.py
├── demo/
│   └── app.py                 # Streamlit demo + monitoring views
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml            # workflow_dispatch
└── docs/                      # PRD / TRD / FLOW / schemas / plan (this documentation pack)
```

## Quickstart (target state when built)

```bash
make setup        # create venv, install pinned deps
make data         # download dataset + write data card (records sha256)
make train        # validate → features → train → log to MLflow → register
make serve        # docker compose up: API on :8000, MLflow UI on :5000
make test         # pytest + lint
make reproduce    # re-run champion pipeline from recorded data hash, diff metrics
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"tenure_months": 12, "monthly_charges": 70.5, "contract": "Month-to-month", ...}'
```

## CV bullets this project earns (once actually built and demoed)

- "Built MLForge, an end-to-end MLOps system (MLflow tracking/registry, FastAPI, Docker, GitHub Actions CI) serving a churn model with data-validation gates and PSI-based drift monitoring; every run reproducible from a recorded data hash via `make reproduce`."
- "Eliminated training/serving skew by shipping a single fitted scikit-learn pipeline as the deployment artefact, with champion/challenger model promotion in the MLflow registry."
- "Implemented automated drift detection comparing live traffic to the training reference distribution, generating drift reports when per-feature PSI exceeded 0.2."

## Status & honesty note

Documentation-first project: PRD, TRD, flows, schemas and implementation plan live in `docs/`. Metrics quoted anywhere in this repo must come from an actual logged MLflow run — no placeholder results.

## Licence

MIT.
