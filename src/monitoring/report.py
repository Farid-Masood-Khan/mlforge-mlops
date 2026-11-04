"""Drift report generation in JSON and Markdown formats per BACKEND_SCHEMA §1.6."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.download import compute_sha256
from src.monitoring.drift import compute_drift_suite


def generate_drift_report(
    logs_file: Path | None = None,
    reference_file: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Reads live prediction JSONL logs and reference snapshot, generating JSON and MD reports."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    logs_file = logs_file or base_dir / "logs" / "predictions.jsonl"
    reference_file = reference_file or base_dir / "data" / "reference" / "holdout_reference.parquet"
    output_dir = output_dir or base_dir / "logs" / "drift_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_file.exists():
        raise FileNotFoundError(f"Reference snapshot missing at {reference_file}")
    if not logs_file.exists():
        raise FileNotFoundError(f"Prediction log missing at {logs_file}. Run simulation/API first.")

    df_reference = pd.read_parquet(reference_file)

    # Read predictions JSONL
    records: list[dict[str, Any]] = []
    with open(logs_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        raise ValueError(f"No valid prediction records found in {logs_file}")

    # Extract current features and probabilities
    cur_features = [r["features"] for r in records]
    cur_probas = [r["probability"] for r in records]
    df_current = pd.DataFrame(cur_features)
    cur_scores = np.array(cur_probas)

    # Compute drift
    drift_result = compute_drift_suite(
        df_reference=df_reference,
        df_current=df_current,
        cur_scores=cur_scores,
    )

    now = datetime.datetime.now(datetime.UTC)
    timestamp_str = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    report_id = f"drift_report_{timestamp_str}"

    raw_csv = base_dir / "data" / "raw" / "telco_churn.csv"
    data_sha = compute_sha256(raw_csv) if raw_csv.exists() else "unknown"

    json_report = {
        "report_id": report_id,
        "generated_at": now.isoformat(),
        "reference": {
            "data_sha256": data_sha,
            "window": "training holdout snapshot",
            "n": len(df_reference),
        },
        "current": {
            "window": "live logged predictions",
            "n": len(df_current),
        },
        "model_version": records[-1].get("model_version", "unknown"),
        "features": drift_result["features"],
        "score_distribution": drift_result["score_distribution"],
        "overall_status": drift_result["overall_status"],
        "thresholds": drift_result["thresholds"],
    }

    # Save JSON report
    json_path = output_dir / f"{report_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Generate Markdown summary
    status_badge = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
    }.get(drift_result["overall_status"], "[INFO]")

    feature_rows = "\n".join(
        [
            f"| `{f['name']}` | `{f['type']}` | {f['psi']:.4f} | {f.get('ks_statistic', '-')} | `{f['status']}` |"
            for f in drift_result["features"]
        ]
    )

    recommendation = (
        "**Retrain Recommended:** Significant feature or score drift detected (PSI > 0.20)."
        if drift_result["overall_status"] == "fail"
        else (
            "**Investigate:** Moderate distribution drift detected (0.10 <= PSI <= 0.20)."
            if drift_result["overall_status"] == "warn"
            else "**No Action Required:** Feature distributions are stable (PSI < 0.10)."
        )
    )

    md_content = f"""# Drift Monitoring Report — {report_id}

- **Generated At:** `{now.isoformat()}`
- **Overall Status:** {status_badge} `{drift_result['overall_status'].upper()}`
- **Reference Records:** `{len(df_reference)}`
- **Analyzed Predictions:** `{len(df_current)}`
- **Recommendation:** {recommendation}

## Feature Drift Summary (PSI & KS)

| Feature | Type | PSI | KS Statistic | Status |
|---|---|---|---|---|
{feature_rows}

## Thresholds
- **Pass:** PSI < 0.10
- **Warn:** 0.10 <= PSI <= 0.20
- **Fail:** PSI > 0.20
"""

    md_path = output_dir / f"{report_id}.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(f"[Monitoring] Drift report generated: {json_path} and {md_path}")
    print(f"[Monitoring] Overall Status: {drift_result['overall_status'].upper()}")

    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drift reports from prediction logs.")
    args = parser.parse_args()
    generate_drift_report()
