# PRD — MLForge End-to-End MLOps Pipeline

**Version:** 1.0 · **Owner:** Farid Masood Khan · **Status:** Pre-build (documentation phase)

## 1. Problem

Data-science portfolios overwhelmingly stop at a notebook with an accuracy number. That artefact demonstrates modelling, but none of the skills that determine whether an ML system survives contact with production:

- **No data contract** — the model silently accepts malformed or shifted inputs.
- **No lineage** — six months later, nobody can say which data, code, or parameters produced the "good" model.
- **No serving story** — the model is a pickle in a folder, not a versioned service.
- **No monitoring** — the model degrades as the input distribution drifts, and nobody notices.

MLForge exists to demonstrate the full lifecycle on one concrete, well-understood problem (customer churn), at a scale one developer can build in ~3 weeks part-time, using the same tooling a real ML team would use.

## 2. Target users

| User | What they need from MLForge |
|---|---|
| **Portfolio reviewers** (professors, scholarship committees, hiring engineers) | In under 10 minutes: understand the system, see it run, verify a run is reproducible, read honest limitations. |
| **ML engineers / data scientists** (the persona the system serves) | Train a model, track the experiment, promote it, serve it, and get alerted when inputs drift — without hand-editing file paths. |
| **The developer (Farid)** | A repeatable template: swap the dataset/config and the same pipeline serves a fraud model or another tabular problem. |

## 3. Goals

1. **G1 — Full lifecycle, one repo.** Raw data → validation → features → training → registry → API → monitoring, all wired together.
2. **G2 — Reproducibility you can test.** Given the recorded data hash + config + seed, re-running training produces metrics within a declared tolerance (±0.005 AUC) of the logged champion run.
3. **G3 — Reviewer-proof demo.** A stranger with Docker can go from `git clone` to a live prediction in under 10 minutes using only the README.
4. **G4 — Honest engineering.** Failures are loud (validation errors, drift reports), limitations are written down, and no metric appears anywhere that did not come from a logged run.

## 4. Non-goals

- **Not** a novel modelling contribution. The model itself is deliberately ordinary (gradient boosting class). The system around it is the contribution.
- **Not** a Kubernetes / cloud deployment. Docker Compose locally is the target; cloud is a stretch note, not a requirement.
- **Not** real-time streaming. Batch scoring endpoint + periodic drift checks are in scope; Kafka-style streaming is not.
- **Not** multi-tenant or user-facing auth. A single static API key is the entire auth model (see TRD).
- **Not** AutoML / hyperparameter search at scale. A small, fixed grid (≤12 configs) logged to MLflow is enough to show tracking value.

## 5. User stories

1. As an ML engineer, I drop a new CSV into `data/raw/`, run `make train`, and the pipeline refuses to train (with a readable error) if a column is missing, a dtype changed, or missing values exceed the schema ceiling.
2. As an ML engineer, I open the MLflow UI and compare all runs — params, AUC/PR-AUC, data hash, git commit — and promote the best run to `champion` with one command.
3. As a consumer of the model, I `POST /predict` with one customer record and get back a churn probability, the binary decision at the registered threshold, and the model version that produced it.
4. As a reviewer, I run `make reproduce` and watch the pipeline retrain from the champion's recorded data hash and print a metric diff against the registry — passing within tolerance.
5. As an ML engineer, after a week of (simulated) live traffic I run the drift job and get a report showing which features drifted (PSI per feature) and whether the score distribution moved.
6. As a reviewer, I open the Streamlit demo, move the tenure/charges sliders, and see the prediction update via the real API — plus a monitoring tab showing the latest drift report.
7. As the developer, every push runs CI: lint, type checks, unit tests, an API test against a tiny fixture model, and a Docker build — a red build blocks merge to `main`.

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Dataset download script writes a **data card**: source URL, row/col counts, sha256 of the file, download date. | Must |
| FR-2 | Validation stage enforces the declared schema (columns, dtypes, value ranges/categories, missing-value ceiling per column, no target column among features). Failure aborts the pipeline with a non-zero exit and a human-readable report. | Must |
| FR-3 | Feature pipeline is a single fitted `sklearn.Pipeline`; the identical object is used in training and serving. | Must |
| FR-4 | Training logs to MLflow: params, config snapshot, seed, data hash, git commit, metrics (ROC-AUC, PR-AUC, F1 at chosen threshold, log loss), and the pipeline artefact. | Must |
| FR-5 | Model registration with `champion`/`challenger` aliases; promotion requires beating the current champion's holdout PR-AUC by ≥0.002 or manual override flag with a logged reason. | Must |
| FR-6 | API endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch` (≤1000 rows), `GET /metrics` (request counts, latency summary, prediction distribution). All requests validated by Pydantic schemas mirroring the data schema. | Must |
| FR-7 | API loads the `champion` model from the registry at startup; model version + alias are returned in every prediction response. | Must |
| FR-8 | Docker: `docker compose up` starts API + MLflow UI with the champion model served; healthcheck on the API container. | Must |
| FR-9 | CI workflow: ruff, mypy, pytest (incl. smoke train on a 200-row fixture), Docker build. Runs on push + PR. | Must |
| FR-10 | Drift job: computes per-feature PSI and prediction-distribution shift vs the reference snapshot; writes a drift report (JSON + one-page markdown) with pass/warn/fail per feature. | Must |
| FR-11 | Retrain workflow (`workflow_dispatch`): runs the full pipeline in CI on the current data drop and uploads the MLflow run summary as a workflow artefact. | Should |
| FR-12 | Streamlit demo: single-record prediction form + monitoring tab rendering the latest drift report. Calls the API only (no direct model loading). | Should |
| FR-13 | Simulated live-traffic generator: samples + perturbs the holdout set to produce "production" batches for drift demonstration. | Should |
| FR-14 | Drift breach opens a GitHub issue automatically (via `gh` in the retrain/drift workflow). | Stretch |
| FR-15 | Feature-store or online/offline split. | Out of scope |

## 7. Success metrics

**System quality**
- Holdout ROC-AUC ≥ 0.80 and PR-AUC ≥ 0.55 on Telco churn with the tuned model (verified from the logged run, not promised in advance — if unmet, the README reports the actual number and why).
- API latency: p95 < 200 ms for single predictions on a laptop-class CPU, model loaded.
- API test coverage: 100% of endpoints covered; `src/` line coverage ≥ 80%.

**Reproducibility criteria (the headline metric)**
- `make reproduce` re-trains from the champion's recorded data hash and config; resulting ROC-AUC and PR-AUC differ from the registered run by ≤ 0.005.
- A clean clone + `make setup && make data && make train` produces a registered model with no manual steps and no undocumented state.
- Every artefact in the registry can be traced to: data hash + git commit + config file + seed, all visible in the MLflow UI.

**Reviewer-experience metrics**
- Clone → first successful `/predict` response: < 10 minutes on a typical laptop with Docker.
- README quickstart commands all succeed verbatim (checked in the final QA pass).

## 8. MVP vs stretch scope

**MVP (must ship):** FR-1…FR-10 — validation, pipeline, MLflow tracking + registry + promotion, FastAPI serving, Docker Compose, CI, drift job + report, reproducible run.

**V1.1 (should ship if time holds):** FR-11…FR-13 — CI retrain workflow, Streamlit demo, traffic simulator.

**Stretch (only after MVP is green):** FR-14 auto-issue on drift; a second dataset (fraud) run through the unchanged pipeline to prove generality; model card generated from the registry; GitHub Pages summary of the champion run.

## 9. Risks & assumptions (summary)

Detailed in IMPLEMENTATION_PLAN.md. Headlines: scope creep toward modelling novelty (guard: model is frozen tech), MLflow-in-Docker networking friction, and dataset licensing (Telco churn is publicly distributed for education; the repo ships a downloader + data card, not the raw file).
