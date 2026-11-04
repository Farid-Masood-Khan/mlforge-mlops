# Data sources — MLForge (MLOps churn pipeline)

Primary dataset: **IBM Telco Customer Churn** (WA_Fn-UseC_Telco-Customer-Churn.csv, ~7,000 rows, 21 fields).
- Download from Kaggle ("Telco Customer Churn" by blastchar) and place the CSV in this folder as `telco_churn.csv`.
- Fallback: generate a synthetic churn dataset with sklearn (`make_classification`) for the first pipeline pass; swap in the real CSV for final runs.
- Known quirk to preserve (do not "clean away"): `total_charges` is blank when `tenure = 0` — see TRD/BACKEND_SCHEMA.
- Never commit datasets to git. Add `*.csv`, `*.parquet` to `.gitignore`.
