"""Unit tests for the PSI calculation and drift detection algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    calculate_categorical_feature_drift,
    calculate_numeric_feature_drift,
    calculate_psi,
    compute_drift_suite,
)


def test_psi_identical_distributions() -> None:
    """Identical distributions produce PSI ~ 0.0."""
    dist = np.array([0.2, 0.3, 0.1, 0.4])
    psi = calculate_psi(dist, dist)
    assert psi < 0.001


def test_psi_shifted_distribution() -> None:
    """Severely shifted distribution produces PSI > 0.20."""
    expected = np.array([0.4, 0.3, 0.2, 0.1])
    actual = np.array([0.05, 0.1, 0.35, 0.5])
    psi = calculate_psi(expected, actual)
    assert psi > 0.20


def test_numeric_feature_drift_unperturbed_and_shifted() -> None:
    """Numeric drift correctly identifies unchanged vs shifted continuous features."""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc=50.0, scale=10.0, size=1000))
    unchanged = pd.Series(rng.normal(loc=50.0, scale=10.0, size=1000))
    shifted = pd.Series(rng.normal(loc=80.0, scale=10.0, size=1000))

    # Unperturbed
    res_unchanged = calculate_numeric_feature_drift(ref, unchanged)
    assert res_unchanged["psi"] < 0.10
    assert res_unchanged["status"] == "pass"

    # Shifted
    res_shifted = calculate_numeric_feature_drift(ref, shifted)
    assert res_shifted["psi"] > 0.20
    assert res_shifted["status"] == "fail"


def test_categorical_feature_drift_detection() -> None:
    """Categorical drift detects share shifts among categories."""
    ref = pd.Series(["A"] * 700 + ["B"] * 300)
    shifted = pd.Series(["A"] * 200 + ["B"] * 800)

    res = calculate_categorical_feature_drift(ref, shifted)
    assert res["psi"] > 0.20
    assert res["status"] == "fail"


def test_compute_drift_suite_overall_status() -> None:
    """Full drift suite flags overall status as fail when a feature fails."""
    df_ref = pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 50,
            "senior_citizen": [0, 1] * 50,
            "monthly_charges": np.linspace(20, 100, 100),
            "total_charges": np.linspace(100, 2000, 100),
            "tenure_months": np.random.randint(0, 72, 100),
            "contract": ["Month-to-month", "One year"] * 50,
        }
    )

    df_cur_shifted = df_ref.copy()
    df_cur_shifted["monthly_charges"] = df_cur_shifted["monthly_charges"] * 2.0  # Massive shift

    report = compute_drift_suite(df_ref, df_cur_shifted)
    assert report["overall_status"] == "fail"
    flagged = [f for f in report["features"] if f["status"] == "fail"]
    assert any(f["name"] == "monthly_charges" for f in flagged)
