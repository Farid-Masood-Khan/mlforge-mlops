"""Feature engineering and preprocessing pipeline.

Constructs unified scikit-learn Pipelines combining median/mode imputation,
scaling, one-hot encoding, and downstream classifiers to eliminate train/serve skew.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Creates unfitted ColumnTransformer per TRD §2.

    - Numerics: median imputation + StandardScaler
    - Categoricals: most-frequent imputation + OneHotEncoder (ignore unknowns)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Assembles full end-to-end Pipeline combining preprocessor and classifier."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "churn",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset into stratified train, val, and test partitions per TRD §2.

    Returns:
        (df_train, df_val, df_test) with original index preserved for reproducibility.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Split ratios must sum to 1.0, got {total_ratio}"

    y = df[target_column]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_val_test = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=y,
    )

    # Second split: val vs test
    val_share = val_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=df_val_test[target_column],
    )

    return df_train, df_val, df_test


def save_reference_snapshot(
    df_holdout: pd.DataFrame,
    reference_dir: Path,
    target_column: str = "churn",
) -> Path:
    """Saves holdout feature snapshot used as reference baseline for drift monitoring."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = reference_dir / "holdout_reference.parquet"

    feature_cols = [c for c in df_holdout.columns if c != target_column]
    df_holdout[feature_cols].to_parquet(snapshot_path, index=False)
    return snapshot_path
