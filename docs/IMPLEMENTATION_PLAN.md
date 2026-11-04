# Implementation Plan — MLForge

**Shape:** ~3 weeks part-time (≈1–1.5 hrs/day weekdays, longer weekend sessions). One developer. Documentation (this pack) already exists — the plan builds the system the docs describe, and any doc/code mismatch found during the build gets fixed in the doc, not silently in the code.

## Ground rules

- **Vertical slices over layers.** Each phase ends with something runnable, not a pile of parts.
- **The model is frozen tech.** LogReg baseline + HistGradientBoosting, small grid. Any urge to "try one more model" is scope creep — log it under Stretch and move on.
- **Commit discipline:** small commits, CI green before moving phases. `main` is never red overnight.

## Phase 0 — Skeleton & tooling (Days 1–2)

- [ ] Repo init from the README structure; `pyproject.toml` (ruff, mypy, pytest), `Makefile` stubs, `.gitignore` (data/raw, mlruns, logs, .env), `.env.example`.
- [ ] `docker-compose.yml` with placeholder API (`/health` returning ok) + MLflow service; verify `docker compose up` works before any real code.
- [ ] CI workflow running lint + an empty pytest suite — green pipeline from day one.
- **Milestone M0:** clone → `docker compose up` → `/health` 200, CI green.

## Phase 1 — Data: ingest + validation gate (Days 3–5)

- [ ] `download.py` + data card (sha256, counts, source).
- [ ] `validate.py` with the full pandera schema from BACKEND_SCHEMA.md, incl. the `total_charges` coercion rule and drop of `customerID`.
- [ ] Tests: valid fixture passes; corrupt fixtures (missing column, wrong dtype, unseen category, excess nulls, duplicate ID) each fail with the right rule named.
- **Milestone M1:** `make data` produces a validated dataset + data card; the bad-data test suite is green.

## Phase 2 — Training, tracking, registry (Days 6–10)

- [ ] Split logic (seeded, stratified, indices persisted).
- [ ] `build.py` preprocessing pipeline; unit tests on transformations (imputation values, one-hot columns, no leakage: fit on train only).
- [ ] `train.py` + `evaluate.py`: baselines + HGB grid (≤12 runs), threshold on validation, single holdout evaluation; full MLflow logging per TRD §3.
- [ ] `promote.py` with the PR-AUC ≥ champion + 0.002 rule and logged override path.
- [ ] Smoke-train test wired into CI (200-row fixture → model registers in a temp store).
- **Milestone M2:** `make train` ends with a registered `champion` v1; MLflow UI shows the full run history with data hashes; holdout metrics recorded in the run summary (actual numbers, whatever they are).

## Phase 3 — Serving (Days 11–14)

- [ ] FastAPI app: lifespan model load from `models:/mlforge-churn@champion`, fail-fast behaviour, `/health` + `/model/info`.
- [ ] Pydantic schemas mirroring the data schema; schema-parity test (imports both, compares fields/types/enums).
- [ ] `/predict` + `/predict/batch` + `/metrics`; prediction JSONL logging with `request_id`.
- [ ] API-key middleware per BACKEND_SCHEMA §3.
- [ ] API tests (TestClient): happy path, 422 on bad category, 401 without key, batch partial errors, 503 path with an empty registry.
- [ ] Dockerfiles finalised (multi-stage, slim); compose healthcheck; latency spot-check vs TRD targets.
- **Milestone M3:** `docker compose up` → `curl /predict` returns a probability with model version + run ID; latency p95 < 200 ms locally.

## Phase 4 — Monitoring & drift (Days 15–17)

- [ ] Reference snapshot generation at promotion time (holdout features + score distribution).
- [ ] `drift.py`: per-feature PSI (numeric: decile bins from reference; categorical: category shares), KS statistic for numerics, score-distribution PSI; thresholds per TRD.
- [ ] `report.py`: JSON + one-page MD report per BACKEND_SCHEMA §1.6.
- [ ] Traffic simulator: perturbed holdout sampling, posts through the API with `source: "simulator"`.
- [ ] Drift maths unit tests: identical distributions → PSI ≈ 0; shifted distribution → PSI > 0.2 on the shifted feature.
- **Milestone M4:** generate 1,000 simulated predictions (500 perturbed) → drift report correctly flags the perturbed features as FAIL/WARN.

## Phase 5 — Demo, retrain workflow, reproducibility (Days 18–21)

- [ ] Streamlit demo per UI_UX_BRIEF (Predict, Monitor, About tabs; API-down and no-report states included).
- [ ] `retrain.yml` workflow_dispatch per TRD §5.
- [ ] `make reproduce`: reads champion's `data_sha256` + config, retrains, prints metric diff, asserts within ±0.005 AUC tolerance.
- [ ] Final QA pass: fresh clone, follow the README quickstart *verbatim*, time it (target < 10 min to first prediction).
- [ ] Write the real Limitations section (below) into the README from what actually happened during the build.
- [ ] Tag `v1.0`; screenshot the demo + MLflow registry for the portfolio/CV.
- **Milestone M5 (Definition of Done for the project):** see checklist below.

## Definition of done

- [ ] All MVP requirements FR-1…FR-10 (PRD §6) demonstrably working.
- [ ] CI green on `main`; smoke-train and bad-data tests present and passing.
- [ ] Reproducibility criterion met and evidenced: `make reproduce` output (metric diff within tolerance) pasted into the run summary.
- [ ] Every number in the README traces to a logged MLflow run.
- [ ] Drift report generated from real logged predictions (simulator traffic labelled as such).
- [ ] Limitations section written and honest.
- [ ] Repo public (or demo-ready), MIT licence, docs pack in `docs/`.

## Testing strategy

| Layer | What | Where |
|---|---|---|
| Unit | Validation rules, PSI maths, threshold logic, transformation expectations | `tests/test_validate.py`, `test_drift.py`, `test_features.py` |
| Contract | API schemas == data schema (fields, types, enums) | `tests/test_schema_parity.py` |
| API | Every endpoint: happy path + error paths (401/422/503), batch partial errors | `tests/test_api.py` |
| End-to-end | Smoke train on fixture → registry contains a model; CI runs it every push | `tests/test_train_smoke.py` |
| Reproducibility | `make reproduce` diff within tolerance (manual/CI-nightly, not per-push — too slow) | Makefile + run summary |
| Manual QA | Fresh-clone quickstart timed; demo states (API down, no report) eyeballed | Phase 5 checklist |

Target: ≥80% line coverage on `src/`, 100% endpoint coverage. Coverage is a floor, not a goal — the four named test files matter more than the percentage.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Scope creep into modelling novelty (XGBoost tuning, feature engineering rabbit holes) | High | Model frozen in PRD; new ideas go to a Stretch list, reviewed only after M5 |
| MLflow + Docker networking/volume friction eats a weekend | Medium | Phase 0 proves compose + MLflow before any real code depends on it |
| PSI binning edge cases (constant features, sparse categories) make drift tests flaky | Medium | Drift maths unit-tested on synthetic distributions first; unseen categories bucketed as `__OTHER__` |
| Telco dataset link rot / licence ambiguity | Low | Downloader + data card documents source; fallback mirror noted in data card; repo ships no raw data |
| Threshold/promotion logic quietly overfits the holdout across many runs | Low–Med | Holdout touched once per run; promotion margin (0.002 PR-AUC) required; noted in Limitations |
| Part-time schedule slips | Medium | Phases are independently demoable — even M3 alone is a presentable project; M4/M5 add the differentiators |

## Planned Limitations section (to be finalised with real build experience)

The README will carry a Limitations section covering, at minimum:

- **Toy scale, real shape.** One small public dataset, one model family, local infrastructure. The patterns transfer; the scale does not, and the project claims patterns, not scale.
- **Demo-grade auth.** A single static API key is not a production security design (no rotation, no per-user identity, no rate limiting).
- **SQLite MLflow backend** will not survive concurrent writers; the tracking URI is config so Postgres is a deployment change, not a code change — but it is untested in this repo.
- **Drift = distribution shift, not performance.** Without live labels, PSI detects that inputs changed, not that accuracy fell. The report says "investigate", never "the model is wrong".
- **Data versioning is hash-based, not DVC.** Traceability is real (every run names its data hash) but there is no dataset lineage graph or remote store.
- **Single holdout reuse.** Repeated promotion decisions against one holdout can overfit it over time; a production system would rotate holdouts or use nested evaluation.
- **No fairness/bias audit** of the churn model across gender/senior-citizen segments — a real deployment would require one; flagged here rather than performed superficially.
