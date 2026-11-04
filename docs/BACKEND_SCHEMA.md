# Backend Schema — MLForge

Data model, storage layout, and the complete API contract. Field names here are canonical: the training schema, Pydantic models, and this document must agree (CI includes a test that imports both and compares them).

## 1. Domain data model

### 1.1 Training record (Telco churn — one row per customer)

Declared in `src/data/validate.py` (pandera) and mirrored in `src/serving/schemas.py`.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `customerID` | string | unique, non-null | **Dropped before features** — identifier, never a feature |
| `gender` | category | `Female`, `Male` | |
| `senior_citizen` | int | 0 or 1 | source CSV uses 0/1 |
| `partner` | category | `Yes`, `No` | |
| `dependents` | category | `Yes`, `No` | |
| `tenure_months` | int | 0–72 | source column `tenure` renamed at ingest |
| `phone_service` | category | `Yes`, `No` | |
| `multiple_lines` | category | `Yes`, `No`, `No phone service` | |
| `internet_service` | category | `DSL`, `Fiber optic`, `No` | |
| `online_security` | category | `Yes`, `No`, `No internet service` | |
| `online_backup` | category | `Yes`, `No`, `No internet service` | |
| `device_protection` | category | `Yes`, `No`, `No internet service` | |
| `tech_support` | category | `Yes`, `No`, `No internet service` | |
| `streaming_tv` | category | `Yes`, `No`, `No internet service` | |
| `streaming_movies` | category | `Yes`, `No`, `No internet service` | |
| `contract` | category | `Month-to-month`, `One year`, `Two year` | |
| `paperless_billing` | category | `Yes`, `No` | |
| `payment_method` | category | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | |
| `monthly_charges` | float | 18.0–120.0 | |
| `total_charges` | float | ≥ 0; may be blank in raw CSV when tenure = 0 → coerced, then median-imputed by the pipeline; raw blank rate > 2% fails validation | Known quirk of this dataset — handled explicitly, documented in the data card |
| `churn` (target) | category | `Yes`, `No` → encoded 1/0 at ingest | Never present in API requests |

**Validation ceilings:** any feature column > 5% missing (after the `total_charges` coercion) fails validation; any unseen category fails validation; duplicate `customerID` fails validation.

### 1.2 Dataset record (one per data drop)

Stored as `data/raw/DATA_CARD.md` + a row in the run's MLflow tags.

| Field | Type | Example |
|---|---|---|
| `data_sha256` | string (hex) | `a1b2c3d4…` |
| `source_url` | string | the public dataset URL used by `download.py` |
| `downloaded_at` | ISO-8601 date | `2026-11-02` |
| `rows` / `cols` | int | 7043 / 21 |
| `target_rate` | float | churn prevalence in the drop (drift context) |

### 1.3 Experiment run (MLflow run — system of record)

| Field | Kind | Notes |
|---|---|---|
| `run_id` | MLflow ID | Referenced by predictions and the registry |
| params: `model_type`, `max_iter`, `learning_rate`, `max_leaf_nodes`, … | params | Full config snapshot also stored as `config_snapshot.yaml` artefact |
| metrics: `roc_auc`, `pr_auc`, `f1`, `log_loss`, `val_roc_auc`, `val_pr_auc` | metrics | Holdout metrics unprefixed; validation prefixed |
| tags: `data_sha256`, `git_commit`, `seed`, `pipeline_version`, `threshold` | tags | `threshold` is the F1-chosen decision threshold, frozen per model version |
| artefacts | files | fitted pipeline (sklearn flavour), `validation_report.json`, `holdout_predictions.parquet` |

### 1.4 Registered model version

MLflow registry, model name `mlforge-churn`: `version` (int, auto), `run_id`, aliases `champion` | `challenger` (at most one version each), tag `former-champion` applied on demotion, tag `promotion_reason` on manual overrides.

### 1.5 Prediction log record (one JSON line per API prediction, `logs/predictions.jsonl`)

```json
{
  "ts": "2026-11-14T10:22:31.412Z",
  "request_id": "b3f1c2…-uuid4",
  "source": "api | demo | simulator",
  "features": { "tenure_months": 12, "monthly_charges": 70.5, "contract": "Month-to-month", "…": "…" },
  "churn_probability": 0.683,
  "churn_prediction": 1,
  "threshold": 0.42,
  "model": { "name": "mlforge-churn", "version": 3, "alias": "champion", "run_id": "8f3a…" },
  "latency_ms": 38
}
```

`source: "simulator"` marks synthetic demo traffic so drift reports can include or exclude it (default: exclude from headline drift numbers, show separately).

### 1.6 Drift report (one JSON + one MD per run, `logs/drift_reports/`)

```json
{
  "report_id": "2026-11-21T09-00-00Z",
  "generated_at": "2026-11-21T09:00:00Z",
  "reference": { "data_sha256": "a1b2c3d4…", "window": "training holdout snapshot", "n": 1057 },
  "current":   { "window": "2026-11-14 → 2026-11-21", "n": 1240, "excluded_simulator": 500 },
  "model_version": 3,
  "features": [
    { "name": "monthly_charges", "type": "numeric", "psi": 0.23, "ks_statistic": 0.11, "status": "fail" },
    { "name": "contract", "type": "categorical", "psi": 0.04, "status": "pass" }
  ],
  "score_distribution": { "psi": 0.16, "reference_mean_prob": 0.27, "current_mean_prob": 0.34, "status": "warn" },
  "overall_status": "fail",
  "thresholds": { "pass_below": 0.1, "fail_above": 0.2 }
}
```

The `.md` sibling is a one-page human summary of the same content (overall status, table of non-pass features, recommendation line: "investigate / retrain / no action").

## 2. API endpoints

Base: `http://localhost:8000` · All endpoints except `/health` require header `X-API-Key`. Content type `application/json` throughout. Errors use FastAPI's standard shape: `{"detail": ...}`.

### `GET /health` — unauthenticated
Response `200`:
```json
{ "status": "ok", "model_loaded": true, "model_version": 3 }
```
`503` with `{"status": "unavailable", "model_loaded": false}` if the registry load failed at startup (the process stays up so the failure is diagnosable, but it does not serve predictions).

### `GET /model/info`
Response `200`:
```json
{
  "name": "mlforge-churn", "version": 3, "alias": "champion", "run_id": "8f3a…",
  "threshold": 0.42, "trained_data_sha256": "a1b2c3d4…",
  "holdout_metrics": { "roc_auc": 0.842, "pr_auc": 0.631, "f1": 0.612 }
}
```
(Metric values above are illustrative shape examples — the live values come from the registry.)

### `POST /predict`
Request — a single training record minus `customerID` and `churn`:
```json
{
  "gender": "Female", "senior_citizen": 0, "partner": "Yes", "dependents": "No",
  "tenure_months": 12, "phone_service": "Yes", "multiple_lines": "No",
  "internet_service": "Fiber optic", "online_security": "No", "online_backup": "Yes",
  "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes",
  "streaming_movies": "No", "contract": "Month-to-month", "paperless_billing": "Yes",
  "payment_method": "Electronic check", "monthly_charges": 70.5, "total_charges": 846.0
}
```
Response `200`:
```json
{
  "churn_probability": 0.683, "churn_prediction": 1, "threshold": 0.42,
  "model": { "name": "mlforge-churn", "version": 3, "alias": "champion", "run_id": "8f3a…" },
  "request_id": "b3f1c2…"
}
```
Errors: `422` field-level validation detail · `401` missing/invalid API key · `503` model not loaded.

### `POST /predict/batch`
Request: `{ "records": [ <record>, … ] }` — 1 to 1000 records.
Response `200`:
```json
{
  "results": [ { "index": 0, "churn_probability": 0.683, "churn_prediction": 1 }, … ],
  "errors":  [ { "index": 7, "detail": "contract: 'Weekly' is not an allowed category" } ],
  "model": { "name": "mlforge-churn", "version": 3, "alias": "champion", "run_id": "8f3a…" }
}
```
Malformed payload (not an object, `records` missing/not a list, >1000 records) → `422` for the whole request.

### `GET /metrics`
Response `200`:
```json
{
  "requests_total": 3120, "errors_total": 41,
  "latency_ms": { "p50": 21, "p95": 87, "p99": 160 },
  "predictions": { "positive_rate": 0.31, "mean_probability": 0.34 },
  "window": "since process start"
}
```

### OpenAPI
FastAPI's auto-generated docs at `/docs` (Swagger UI) and `/openapi.json` are the live contract; this file is the human-readable mirror. If they disagree, `/openapi.json` wins and this file gets fixed.

## 3. Auth model

- **Scheme:** static API key, header `X-API-Key`, compared with `hmac.compare_digest` against env var `MLFORGE_API_KEY`.
- **Scope:** one key, read+predict (no admin endpoints exist). `/health` exempt (needed by Docker healthcheck and load balancers).
- **Demo app** reads the same key from its own env; the key never appears in code, logs, or the Streamlit UI.
- **Explicitly demo-grade:** no users, no rotation, no rate limiting in MVP. The TRD and README both label it as such — a reviewer should never mistake this for a production auth design.
