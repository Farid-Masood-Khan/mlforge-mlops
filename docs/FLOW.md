# FLOW — MLForge

Four flows. Diagrams are Mermaid (render on GitHub). "Gate" steps can stop the flow — that is by design.

## 1. Training flow

Runs via `make train` (or the training container). One pass = one MLflow run per model config.

```mermaid
flowchart TD
    A[Download dataset<br/>download.py] --> B[Write data card<br/>sha256, rows, source]
    B --> C{Schema validation<br/>validate.py}
    C -- FAIL --> X1[Abort: print all<br/>violated rules, exit 1]
    C -- PASS --> D[Stratified split 70/15/15<br/>seeded, indices stored]
    D --> E[Build preprocessing<br/>Pipeline - build.py]
    E --> F[Train candidate models<br/>LogReg baseline + HGB grid ≤12 runs]
    F --> G[Evaluate on validation<br/>choose threshold by F1]
    G --> H[Final metrics on holdout<br/>ROC-AUC, PR-AUC, F1, log loss]
    H --> I[Log run to MLflow<br/>params, metrics, data hash,<br/>git commit, artefact]
    I --> J{Promotion check<br/>PR-AUC ≥ champion + 0.002?}
    J -- YES --> K[Register version<br/>alias: champion<br/>old champion → archived tag]
    J -- NO --> L[Register version<br/>alias: challenger]
    K --> M[Write run summary<br/>to docs/runs/ + console]
    L --> M
```

Key properties: the holdout is touched exactly once per run (step H); the threshold is chosen on validation (step G) *before* that. Validation failure is the only hard gate before training; promotion is the only gate after it.

## 2. Inference flow

```mermaid
sequenceDiagram
    participant C as Client / Streamlit demo
    participant A as FastAPI app
    participant V as Pydantic schema
    participant P as Predictor (pipeline artefact)
    participant R as MLflow Registry
    participant L as Prediction log (JSONL)

    Note over A,R: At startup: load models:/mlforge-churn@champion, cache version + run_id
    C->>A: POST /predict  {customer features, X-API-Key}
    A->>V: Validate + coerce types
    alt invalid request
        V-->>A: field errors
        A-->>C: 422 {detail: [...]}
    else valid
        V-->>A: clean record (training column order)
        A->>P: predict_proba(record)
        P-->>A: probability
        A->>A: apply registered threshold → decision
        A->>L: append {ts, request_id, features, prob, decision, model_version}
        A-->>C: 200 {churn_probability, churn_prediction, threshold, model{version, alias, run_id}}
    end
```

Batch (`/predict/batch`) follows the same path per row; row-level validation failures are collected into an `errors` array instead of failing the request.

## 3. CI flow

Runs on every push and pull request to `main` (GitHub Actions, `ci.yml`).

```mermaid
flowchart LR
    A[git push / PR] --> B[Lint<br/>ruff check + format]
    B --> C[Type check<br/>mypy src/]
    C --> D[Unit tests<br/>validate, features, drift maths]
    D --> E[API tests<br/>TestClient + fixture model]
    E --> F[Smoke train<br/>200-row fixture → temp MLflow<br/>assert model registers]
    F --> G[Bad-data test<br/>corrupt CSV must<br/>FAIL validation]
    G --> H[Docker build<br/>api + train images]
    H --> I{All green?}
    I -- YES --> J[Merge allowed<br/>branch protection]
    I -- NO --> K[Build red<br/>merge blocked]
```

The two tests reviewers should notice: **F** proves the whole pipeline runs end-to-end on every commit, and **G** proves the validation gate actually rejects bad data (a gate that never fires in tests is decoration).

## 4. Retraining flow

Two triggers: manual (`make train` on a new data drop, or the `retrain` GitHub workflow) and drift-driven (a human decision after reading a drift report — automation proposes, the promotion rule disposes).

```mermaid
flowchart TD
    A{Trigger} -->|New data drop| B[Place CSV in data/raw/]
    A -->|Drift report: ≥1 feature FAIL| B2[Reviewer/ML engineer<br/>inspects drift report]
    B2 --> B
    A -->|workflow_dispatch| B
    B --> C[Training flow<br/>see FLOW §1]
    C --> D{New run beats champion?<br/>holdout PR-AUC ≥ champion + 0.002}
    D -- YES --> E[Alias swap: new version → champion<br/>previous champion keeps version,<br/>tagged 'former-champion']
    D -- NO --> F[New version stays challenger<br/>run summary records why]
    E --> G[Restart/reload API<br/>docker compose restart api<br/>startup loads new champion]
    F --> H[No serving change]
    G --> I[Smoke check: GET /model/info<br/>shows new version; POST /predict OK]
    I --> J[Regenerate reference snapshot?<br/>Only by explicit decision —<br/>recorded in drift report history]
```

Two deliberate design choices:

1. **No automatic promotion.** Drift can *trigger* retraining, but a model only becomes `champion` by beating the incumbent on a held-out set. Auto-promoting on drift alone would let a shifted data drop replace a good model with an unproven one.
2. **The reference snapshot is not auto-updated.** If the drift baseline silently moved with the data, drift detection would grade its own homework. Rebaselining is an explicit, recorded decision (step J).
