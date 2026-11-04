# TRD — MLForge End-to-End MLOps Pipeline

**Version:** 1.0 · **Companion docs:** PRD.md (scope), BACKEND_SCHEMA.md (data/API contracts), FLOW.md (diagrams)

## 1. System architecture

Five components, deliberately boring technology, one responsibility each:

```
┌────────────┐   CSV drop    ┌──────────────┐   fitted pipeline   ┌──────────────┐
│  Data      │ ────────────► │  Training    │ ──────────────────► │  MLflow      │
│  source    │               │  pipeline    │   logs + artefact   │  Tracking +  │
│ (download  │               │  validate →  │                     │  Registry    │
│  script)   │               │  features →  │                     │  (SQLite +   │
└────────────┘               │  train →     │                     │  artefacts)  │
                             │  evaluate →  │                     └──────┬───────┘
                             │  promote     │                            │ champion alias
                             └──────────────┘                            ▼
┌────────────┐   samples     ┌──────────────┐   HTTP/JSON    ┌──────────────┐
│  Traffic   │ ────────────► │  Prediction  │ ◄───────────── │  FastAPI     │
│  simulator │               │  log (JSONL) │                │  service     │
└────────────┘               └──────┬───────┘                └──────────────┘
                                    │                                ▲
                                    ▼                                │ HTTP
                             ┌──────────────┐                ┌──────────────┐
                             │  Drift job   │ ─── report ──► │  Streamlit   │
                             │  (PSI vs     │   (JSON + MD)  │  demo /      │
                             │  reference)  │                │  monitoring  │
                             └──────────────┘                └──────────────┘
```

Deployment unit: `docker-compose.yml` with two services — `api` (FastAPI) and `mlflow` (tracking server UI) — sharing a named volume for the artefact store. Training and drift run as one-off containers (`docker compose run train`) or directly via `make` in a venv during development.

## 2. Pipeline stages

| Stage | Module | Input | Output | Failure behaviour |
|---|---|---|---|---|
| 0. Ingest | `src/data/download.py` | Source URL (config) | `data/raw/*.csv` + `data/raw/DATA_CARD.md` (sha256, rows, cols, source, date) | Non-zero exit if hash of re-download differs and `--force` not passed |
| 1. Validate | `src/data/validate.py` | Raw CSV + `schema.py` | Validated DataFrame or failure report | **Hard fail.** Prints every violated rule; pipeline aborts before any fitting |
| 2. Features | `src/features/build.py` | Validated DataFrame | Unfitted `sklearn.Pipeline` (ColumnTransformer: median impute + StandardScaler for numerics; most-frequent impute + OneHotEncoder for categoricals; then model) | Unit-tested transformation expectations |
| 3. Train | `src/models/train.py` | Pipeline + train split | Fitted pipeline, MLflow run | Trains baseline (LogReg) and main model (HistGradientBoosting); small fixed grid ≤12 configs via separate runs |
| 4. Evaluate | `src/models/evaluate.py` | Fitted pipeline + holdout | Metrics dict; threshold chosen on validation split to maximise F1, frozen into run | Holdout never touched during threshold selection |
| 5. Register/Promote | `src/models/promote.py` | Run ID + holdout metrics | Model version in registry; alias assignment | Promotion rule: PR-AUC ≥ champion + 0.002, else stays `challenger`; override requires `--force --reason "..."` logged as a run tag |
| 6. Serve | `src/serving/` | Registry alias `champion` | HTTP predictions | Startup fails fast if registry unreachable or alias missing — no silent fallback model |
| 7. Monitor | `src/monitoring/` | Prediction log + `data/reference/` snapshot | Drift report JSON + MD | Thresholds: PSI < 0.1 pass, 0.1–0.2 warn, > 0.2 fail |

Split discipline: stratified 70/15/15 train/val/holdout, split indices derived from the global seed and stored with the run, so the holdout is identical across runs on the same data hash.

## 3. MLflow setup

- **Backend store:** SQLite (`mlflow.db`) — adequate for single-developer scale, zero extra services. Path parameterised by `MLFLOW_TRACKING_URI` so a Postgres URI can replace it without code changes.
- **Artefact store:** local directory (`./mlartifacts`) in dev; named Docker volume in Compose.
- **Experiment:** single experiment `mlforge-churn`. Every run logs:
  - params: model type, hyperparameters, feature-set version, threshold;
  - metrics: `roc_auc`, `pr_auc`, `f1`, `log_loss` (holdout), plus validation metrics prefixed `val_`;
  - tags: `data_sha256`, `git_commit`, `config_file`, `seed`, `pipeline_version`;
  - artefacts: the fitted pipeline (via `mlflow.sklearn.log_model`), `validation_report.json`, `holdout_predictions.parquet` (for error analysis).
- **Registry:** model name `mlforge-churn`. Aliases `champion` (served) and `challenger` (latest candidate). Stage transitions are not used (deprecated in MLflow ≥2.x in favour of aliases — the design follows current practice).
- **Serving load:** `mlflow.sklearn.load_model("models:/mlforge-churn@champion")` at API startup; version + run ID cached and exposed at `/model/info`.

## 4. Serving design

- **Framework:** FastAPI + uvicorn (2 workers in Docker). App factory in `src/serving/app.py`; model loaded once in the lifespan handler, not per-request.
- **Validation:** Pydantic request models generated to mirror the training schema (same field names, types, category enums). A request that would fail data validation gets a `422` with the field-level error — the API never passes unvalidated data to the model.
- **Prediction path:** request → Pydantic model → single-row DataFrame with training column order → pipeline `predict_proba` → response `{churn_probability, churn_prediction, threshold, model: {name, version, alias, run_id}}`.
- **Prediction log:** every prediction is appended as one JSON line (`timestamp, request_id, features, probability, prediction, model_version`) to `logs/predictions.jsonl` — this is the drift job's input and the audit trail. Log rotation by size (10 MB) via `RotatingFileHandler`-equivalent in the logging config.
- **Batch:** `/predict/batch` accepts ≤1000 records, returns per-row results; partial validation failures return per-row errors rather than failing the whole batch (`200` with an `errors` array; `422` only if the payload itself is malformed).
- **Metrics endpoint:** in-process counters (requests, errors, latency histogram buckets) summarised at `/metrics` as JSON (Prometheus exporter is a stretch item; JSON keeps the MVP dependency-light).

## 5. CI/CD design

**`ci.yml`** (on push + PR to `main`):
1. Checkout, Python 3.11, cache pip.
2. `ruff check` + `ruff format --check`.
3. `mypy src/` (strict on `serving/`, lenient elsewhere — configured in `pyproject.toml`).
4. `pytest` — unit tests + API tests + **smoke train**: trains on a 200-row fixture in a temp MLflow store and asserts a model registers.
5. Data-validation test: the fixture "bad CSV" (missing column, wrong dtype, excess nulls) must fail validation — a test that the gate actually gates.
6. `docker build` for both Dockerfiles (build only; no push).

**`retrain.yml`** (`workflow_dispatch` with inputs: `data_path` note, `force_promote` boolean):
1. Runs stages 0–5 against a local file-based MLflow store.
2. Uploads run summary (metrics table + validation report) as a workflow artefact.
3. Promotion still obeys the PR-AUC rule unless `force_promote` is set — CI never silently replaces the champion.

**Branch protection (documented, manual step):** `main` requires the CI workflow green. No CD in MVP — "deployment" is `docker compose up` from a tagged commit, and the README says so honestly.

## 6. Tech choices & rationale

| Decision | Chosen | Alternative considered | Rationale |
|---|---|---|---|
| Tracking/registry | MLflow | Weights & Biases, DVC-only | Free, self-hostable, registry + aliases built in, the tool reviewers most recognise |
| Validation | pandera | Great Expectations | GE is powerful but heavyweight for one dataset; pandera schemas are readable in a code review |
| Serving | FastAPI | Flask, MLflow built-in serving | Typed requests catch schema drift at the boundary; auto OpenAPI docs help reviewers |
| Model | HistGradientBoosting (+ LogReg baseline) | XGBoost/LightGBM first | Zero extra deps for a strong tabular model; XGBoost is a config-level swap shown in the grid |
| Backend store | SQLite | Postgres | Single-dev scale; URI is config, so the upgrade path is one line |
| Data versioning | Content hash + data card | Full DVC | Honest middle ground: the hash gives traceability without DVC's remote-storage ceremony at this scale; the TRD names this trade-off explicitly |
| Demo UI | Streamlit calling the API | Custom frontend | Keeps the demo honest (it exercises the real service) with minimal code |
| Drift metric | PSI per feature + score-distribution PSI | Evidently, KS tests everywhere | PSI is the industry-common language for tabular drift; KS statistic also logged for numeric features as a secondary signal |

## 7. Security requirements

- **Auth:** single static API key via `X-API-Key` header, read from env var `MLFORGE_API_KEY`; `/health` is unauthenticated, everything else requires the key. Constant-time comparison (`hmac.compare_digest`). This is demo-grade auth and the docs say so — no user accounts, no OAuth in scope.
- **Secrets:** none in the repo. `.env.example` documents required vars; `.env` is git-ignored; CI uses a throwaway test key.
- **Input safety:** request size caps (batch ≤1000 rows, body ≤2 MB); category enums reject unseen values with `422` rather than passing them to the encoder; no `eval`/pickle loading outside MLflow's artefact handling.
- **Dependencies:** pinned versions; `pip-audit` step in CI (warn-only in MVP).
- **Data:** public Telco dataset, no PII beyond the dataset's own synthetic customer IDs, which are dropped at validation (`customerID` is in the schema's drop-list).

## 8. Performance requirements

- Single-prediction latency p95 < 200 ms on laptop CPU (model warm); batch of 1000 < 3 s.
- API memory footprint < 512 MB with the model loaded.
- Training run (full Telco dataset, main model) < 2 minutes on laptop CPU — keeps `make reproduce` cheap enough that reviewers will actually run it.
- Drift job over 10k logged predictions < 30 s.
- Docker images: API image < 600 MB (multi-stage build, slim base).
