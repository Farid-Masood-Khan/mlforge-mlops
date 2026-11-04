"""Model evaluation and decision threshold optimization module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def optimize_threshold(
    pipeline: Pipeline,
    df_val: pd.DataFrame,
    target_column: str = "churn",
    metric: str = "f1",
) -> float:
    """Finds optimal binary decision threshold on the validation set.

    Grid search over [0.10, 0.90] to maximize target metric (default F1).
    Frozen before evaluating on holdout to prevent data snooping.
    """
    feature_cols = [c for c in df_val.columns if c != target_column]
    x_val = df_val[feature_cols]
    y_val = df_val[target_column].to_numpy()

    y_proba = pipeline.predict_proba(x_val)[:, 1]

    best_threshold = 0.50
    best_score = -1.0

    thresholds = np.linspace(0.10, 0.90, 81)
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        if metric == "f1":
            score = float(f1_score(y_val, preds, zero_division=0))
        else:
            score = float(f1_score(y_val, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return round(best_threshold, 4)


def evaluate_split(
    pipeline: Pipeline,
    df: pd.DataFrame,
    threshold: float = 0.50,
    target_column: str = "churn",
) -> dict[str, float]:
    """Evaluates fitted pipeline against a dataset split with a fixed threshold."""
    feature_cols = [c for c in df.columns if c != target_column]
    x_split = df[feature_cols]
    y_true = df[target_column].to_numpy()

    y_proba = pipeline.predict_proba(x_split)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    loss = float(log_loss(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
    }


def create_error_analysis_dataframe(
    pipeline: Pipeline,
    df_holdout: pd.DataFrame,
    threshold: float,
    target_column: str = "churn",
) -> pd.DataFrame:
    """Generates holdout predictions dataframe with residual errors for error analysis."""
    feature_cols = [c for c in df_holdout.columns if c != target_column]
    x_holdout = df_holdout[feature_cols]
    y_true = df_holdout[target_column].to_numpy()

    y_proba = pipeline.predict_proba(x_holdout)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    analysis_df = df_holdout.copy()
    analysis_df["true_churn"] = y_true
    analysis_df["predicted_proba"] = np.round(y_proba, 4)
    analysis_df["predicted_churn"] = y_pred
    analysis_df["is_error"] = (y_true != y_pred).astype(int)
    return analysis_df
