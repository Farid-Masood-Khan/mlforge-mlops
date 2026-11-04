"""Population Stability Index (PSI) and distribution drift monitoring engine.

Calculates per-feature PSI, Kolmogorov-Smirnov (KS) statistic, and model score PSI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.data.validate import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPSILON = 1e-4


def calculate_psi(expected_shares: np.ndarray, actual_shares: np.ndarray) -> float:
    """Calculates Population Stability Index between expected and actual proportions."""
    # Apply Laplace smoothing to avoid division by zero or log(0)
    expected_shares = np.clip(expected_shares, EPSILON, 1.0)
    actual_shares = np.clip(actual_shares, EPSILON, 1.0)

    # Re-normalize
    expected_shares = expected_shares / np.sum(expected_shares)
    actual_shares = actual_shares / np.sum(actual_shares)

    psi_value = np.sum((actual_shares - expected_shares) * np.log(actual_shares / expected_shares))
    return float(np.round(psi_value, 4))


def calculate_numeric_feature_drift(
    ref_series: pd.Series,
    cur_series: pd.Series,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Calculates PSI and KS statistic for continuous numeric features using reference decile bins."""
    ref_clean = ref_series.dropna().to_numpy()
    cur_clean = cur_series.dropna().to_numpy()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {"psi": 0.0, "ks_statistic": 0.0, "status": "pass"}

    # Compute reference bin edges (deciles)
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # Handle duplicate quantile edges

    if len(bin_edges) <= 1:
        bin_edges = np.array([ref_clean.min() - 1e-3, ref_clean.max() + 1e-3])

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_shares = ref_counts / len(ref_clean)
    cur_shares = cur_counts / len(cur_clean)

    psi = calculate_psi(ref_shares, cur_shares)
    ks_res = stats.ks_2samp(ref_clean, cur_clean)
    ks_stat = float(np.round(ks_res.statistic, 4))

    status = "fail" if psi > 0.20 else ("warn" if psi >= 0.10 else "pass")

    return {
        "psi": psi,
        "ks_statistic": ks_stat,
        "status": status,
    }


def calculate_categorical_feature_drift(
    ref_series: pd.Series,
    cur_series: pd.Series,
) -> dict[str, Any]:
    """Calculates PSI for categorical features using relative category distributions."""
    ref_clean = ref_series.dropna().astype(str)
    cur_clean = cur_series.dropna().astype(str)

    all_categories = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))

    ref_shares = np.array([(ref_clean == cat).mean() for cat in all_categories])
    cur_shares = np.array([(cur_clean == cat).mean() for cat in all_categories])

    psi = calculate_psi(ref_shares, cur_shares)
    status = "fail" if psi > 0.20 else ("warn" if psi >= 0.10 else "pass")

    return {
        "psi": psi,
        "status": status,
    }


def compute_drift_suite(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    ref_scores: np.ndarray | None = None,
    cur_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Computes comprehensive feature and score drift analysis."""
    features_report: list[dict[str, Any]] = []
    overall_status = "pass"

    # Numeric features drift
    for col in NUMERIC_FEATURES:
        if col in df_reference.columns and col in df_current.columns:
            res = calculate_numeric_feature_drift(df_reference[col], df_current[col])
            features_report.append(
                {
                    "name": col,
                    "type": "numeric",
                    "psi": res["psi"],
                    "ks_statistic": res["ks_statistic"],
                    "status": res["status"],
                }
            )
            if res["status"] == "fail":
                overall_status = "fail"
            elif res["status"] == "warn" and overall_status != "fail":
                overall_status = "warn"

    # Categorical features drift
    for col in CATEGORICAL_FEATURES:
        if col in df_reference.columns and col in df_current.columns:
            res = calculate_categorical_feature_drift(df_reference[col], df_current[col])
            features_report.append(
                {
                    "name": col,
                    "type": "categorical",
                    "psi": res["psi"],
                    "status": res["status"],
                }
            )
            if res["status"] == "fail":
                overall_status = "fail"
            elif res["status"] == "warn" and overall_status != "fail":
                overall_status = "warn"

    # Score distribution drift
    score_report: dict[str, Any] = {}
    if (
        ref_scores is not None
        and cur_scores is not None
        and len(ref_scores) > 0
        and len(cur_scores) > 0
    ):
        score_res = calculate_numeric_feature_drift(pd.Series(ref_scores), pd.Series(cur_scores))
        score_report = {
            "psi": score_res["psi"],
            "reference_mean_prob": round(float(np.mean(ref_scores)), 4),
            "current_mean_prob": round(float(np.mean(cur_scores)), 4),
            "status": score_res["status"],
        }
        if score_res["status"] == "fail":
            overall_status = "fail"
        elif score_res["status"] == "warn" and overall_status != "fail":
            overall_status = "warn"

    return {
        "features": features_report,
        "score_distribution": score_report,
        "overall_status": overall_status,
        "thresholds": {"pass_below": 0.10, "fail_above": 0.20},
    }
