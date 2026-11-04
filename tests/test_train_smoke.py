"""Smoke test for MLForge training pipeline and MLflow registry.

Trains a fast baseline on a 200-row fixture in a temporary MLflow store to verify
lineage tracking, artifact generation, and alias assignment.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build import build_pipeline, split_dataset
from src.models.evaluate import evaluate_split, optimize_threshold
from src.models.promote import evaluate_and_promote_candidate


@pytest.fixture
def fixture_200_rows() -> pd.DataFrame:
    """Generates 200 rows of valid synthetic data for rapid CI smoke testing."""
    rng = np.random.default_rng(101)
    n = 200
    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], size=n),
            "senior_citizen": rng.choice([0, 1], size=n),
            "partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "tenure_months": rng.integers(0, 72, size=n),
            "phone_service": rng.choice(["Yes", "No"], size=n),
            "multiple_lines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internet_service": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "online_security": rng.choice(["Yes", "No", "No internet service"], size=n),
            "online_backup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "device_protection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "tech_support": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streaming_tv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streaming_movies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "paperless_billing": rng.choice(["Yes", "No"], size=n),
            "payment_method": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "monthly_charges": rng.uniform(20.0, 110.0, size=n),
            "total_charges": rng.uniform(20.0, 5000.0, size=n),
            "churn": rng.choice([0, 1], p=[0.70, 0.30], size=n),
        }
    )


def test_smoke_train_and_register(tmp_path: Path, fixture_200_rows: pd.DataFrame) -> None:
    """Verifies end-to-end training, threshold selection, and registry promotion."""
    db_path = tmp_path / "temp_mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts"
    artifacts_dir.mkdir()

    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "test-smoke-experiment"
    mlflow.set_experiment(experiment_name)

    df_train, df_val, df_holdout = split_dataset(
        fixture_200_rows,
        target_column="churn",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    feature_cols = [c for c in fixture_200_rows.columns if c != "churn"]
    x_train, y_train = df_train[feature_cols], df_train["churn"].to_numpy()

    with mlflow.start_run(run_name="smoke-run") as run:
        pipeline = build_pipeline(LogisticRegression(max_iter=100))
        pipeline.fit(x_train, y_train)

        thresh = optimize_threshold(pipeline, df_val, target_column="churn")
        holdout_metrics = evaluate_split(
            pipeline,
            df_holdout,
            threshold=thresh,
            target_column="churn",
        )

        mlflow.log_params({"threshold": thresh, "model": "smoke_logreg"})
        for k, v in holdout_metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    # Promote in temporary registry
    assigned_alias, msg = evaluate_and_promote_candidate(
        run_id=run_id,
        candidate_pr_auc=holdout_metrics["pr_auc"],
        model_name="test-churn-model",
    )

    assert assigned_alias == "champion"
    assert "promoted to 'champion'" in msg
