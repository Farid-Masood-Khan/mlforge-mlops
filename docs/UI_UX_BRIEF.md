# UI/UX Brief — MLForge Demo & Monitoring Surface

MLForge is a backend project. Its "UI" has one job: **let a reviewer verify, in under five minutes, that the system is real** — a live model behind an API, experiments tracked, and drift being watched. Two surfaces do this: the **Streamlit demo app** (`demo/app.py`, a thin client that only talks to the FastAPI service) and the **MLflow UI** (stock, on `:5000`). No custom frontend is built; this brief specifies what the reviewer sees and in what order.

## Design principles

1. **Proof over polish.** Every screen shows something a notebook can't fake: a model version + run ID from the live registry, a latency reading, a drift table computed from logged predictions.
2. **The demo must exercise the real system.** The Streamlit app never loads a model directly. If the API is down, the demo says so — it does not fall back to a local pickle.
3. **Honest states are UI states.** "No drift report yet", "API unreachable", "model is the baseline" are designed screens, not accidents.

## Surface 1 — Streamlit demo (port 8501)

### Screen A: Predict (default tab)

**Purpose:** one prediction, end to end, with provenance.

Layout:
- Left column — input form mirroring the API schema exactly: `tenure_months` (slider 0–72), `monthly_charges` (slider 18–120), `total_charges` (number), `contract` (select: Month-to-month / One year / Two year), `internet_service` (DSL / Fiber optic / No), `payment_method` (select, 4 options), `paperless_billing` (toggle), `senior_citizen` (toggle), plus the remaining categorical selects in an "More fields" expander pre-filled with the dataset's modal values.
- Right column — result card (after **Predict** is clicked):
  - Churn probability as a percentage + a horizontal bar against the registered decision threshold marker.
  - Decision chip: `Likely to churn` / `Likely to stay` (colour: amber/green — never red/green alone, colour-blind safe pairing with the text label).
  - Provenance line (small, monospace): `model mlforge-churn v3 · alias champion · run 8f3a… · threshold 0.42 · latency 38 ms`. This line is the point of the screen.
- Footer note: "Predictions are logged and used by the drift monitor (tab 2)."

States:
- **API unreachable:** full-width warning — "The prediction API isn't responding. Start it with `make serve`, then retry." + a Retry button. No fake result is ever rendered.
- **Validation error (422):** the specific field is highlighted under the form with the API's error text (should be unreachable given the form mirrors the schema — if it happens, that's a bug the UI surfaces honestly).
- **Slow response (>2 s):** spinner with "Scoring…" — and the eventual latency is still shown, even if embarrassing.

### Screen B: Monitor (tab 2)

**Purpose:** show the latest drift report and the model's recent behaviour.

Layout:
- Header strip: last drift-run timestamp, overall status chip (`PASS` / `WARN` / `FAIL`), predictions analysed (n), reference window description.
- Drift table: one row per feature — feature name, type, PSI value, status chip per thresholds (<0.1 pass, 0.1–0.2 warn, >0.2 fail), and a sparkline-style bar of reference vs current distribution for the selected row.
- Score-distribution panel: histogram overlay (reference vs live prediction probabilities) + prediction-rate (mean predicted churn) for both windows.
- Run history: small table of past drift reports (date, status, worst feature) read from `logs/drift_reports/`.
- "Run drift check now" button (calls the drift job against the current prediction log; disabled with an explanatory caption when fewer than 200 predictions are logged — PSI on tiny samples is noise, and the UI says why).

States:
- **No report exists yet:** empty-state panel explaining how to produce one (generate demo traffic → run drift check) with a button to generate 500 simulated predictions through the API (uses the traffic simulator; labels them clearly as simulated in the log).
- **Stale report (>7 days or >5,000 predictions since):** amber banner — "This report predates the last N predictions."

### Screen C: About / How to verify (tab 3, static)

A short page that tells the reviewer exactly how to check the project's claims: the `make reproduce` command and what passing looks like, where the champion run lives in MLflow, where the data card is, and a link list (PRD, TRD, schema, limitations). This tab exists because reviewers reward projects that make verification easy.

## Surface 2 — MLflow UI (stock, port 5000)

No customisation beyond organisation:
- One experiment, `mlforge-churn`; runs named by `model_type + data hash prefix` (e.g. `hgb_a1b2c3d4`).
- The **Registered Models** view is the reviewer's destination: `mlforge-churn` with versions and the `champion`/`challenger` aliases visible.
- README links directly to the experiment view and explains the three tags to filter by (`data_sha256`, `git_commit`, `seed`).
- The demo's provenance line gives the run ID so a reviewer can go from a prediction → the exact run → its params, metrics, and data hash in two clicks.

## Reviewer journey (the 5-minute path the UX is designed around)

1. `docker compose up` → open Streamlit (0:00–1:00).
2. Make a prediction; notice the provenance line (1:00–2:00).
3. Switch to Monitor tab; read the drift table (2:00–3:00).
4. Click through to MLflow; find the run ID from step 2 (3:00–4:00).
5. Read the About tab; run `make reproduce` (4:00–5:00, command runs longer — that's fine, the claim is now credible).

## Non-requirements

- No user accounts, no theming/branding work, no mobile responsiveness beyond "doesn't break".
- No charts library beyond Streamlit's built-ins + one histogram — visual restraint is deliberate; this is an engineering artefact, not a product pitch.
