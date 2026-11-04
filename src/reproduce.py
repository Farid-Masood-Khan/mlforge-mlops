"""Reproducibility verification script for MLForge.

Re-runs training pipeline from data hash and asserts that champion holdout metrics
match previous recorded metrics within float tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.models.train import run_training_pipeline


def verify_reproducibility(tolerance: float = 0.005) -> bool:
    """Re-executes the pipeline and compares holdout metrics with recorded numbers."""
    base_dir = Path(__file__).resolve().parent.parent
    summary_path = base_dir / "results" / "metrics_summary.json"

    if not summary_path.exists():
        print(f"[Reproduce] No baseline summary found at {summary_path}. Run training first.")
        return False

    with open(summary_path, encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_runs = {r["name"]: r for r in baseline["runs"]}
    print("[Reproduce] Running training pipeline to verify reproducibility...")

    # Run training
    new_result = run_training_pipeline()
    new_runs = {r["name"]: r for r in new_result["runs_summary"]}

    print("\n[Reproduce] Metric Verification Audit:")
    all_matched = True

    for name, orig_run in baseline_runs.items():
        if name not in new_runs:
            print(f"  [MISSING] Run '{name}' not found in new execution.")
            all_matched = False
            continue

        reproduced_run = new_runs[name]
        orig_pr_auc = float(orig_run["holdout_pr_auc"])
        new_pr_auc = float(reproduced_run["holdout_pr_auc"])
        diff = abs(orig_pr_auc - new_pr_auc)

        if diff <= tolerance:
            print(
                f"  [PASS] {name}: Baseline PR-AUC={orig_pr_auc:.4f}, Reproduced={new_pr_auc:.4f} (diff={diff:.5f})"
            )
        else:
            print(
                f"  [FAIL] {name}: Baseline PR-AUC={orig_pr_auc:.4f}, Reproduced={new_pr_auc:.4f} (diff={diff:.5f} > tol {tolerance})"
            )
            all_matched = False

    if all_matched:
        print("\n[Reproduce] SUCCESS: All model metrics reproduced within tolerance!")
    else:
        print("\n[Reproduce] WARNING: One or more metric comparisons exceeded tolerance.")

    return all_matched


if __name__ == "__main__":
    success = verify_reproducibility()
    sys.exit(0 if success else 1)
