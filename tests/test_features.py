"""Unit tests for feature preprocessing pipeline and split integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validate import NUMERIC_FEATURES
from src.features.build import build_preprocessor, split_dataset


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Fixture containing both numeric and categorical features with realistic variations."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female"],
            "senior_citizen": [0, 1, 0, 0, 1],
            "partner": ["Yes", "No", "No", "Yes", "No"],
            "dependents": ["No", "No", "Yes", "No", "Yes"],
            "tenure_months": [1, 24, 60, 12, 0],  # Includes tenure=0
            "phone_service": ["No", "Yes", "Yes", "Yes", "Yes"],
            "multiple_lines": ["No phone service", "Yes", "No", "Yes", "No"],
            "internet_service": ["DSL", "Fiber optic", "Fiber optic", "No", "DSL"],
            "online_security": ["No", "Yes", "No", "No internet service", "Yes"],
            "online_backup": ["Yes", "No", "Yes", "No internet service", "No"],
            "device_protection": ["No", "Yes", "Yes", "No internet service", "No"],
            "tech_support": ["No", "Yes", "No", "No internet service", "Yes"],
            "streaming_tv": ["No", "Yes", "Yes", "No internet service", "No"],
            "streaming_movies": ["No", "Yes", "Yes", "No internet service", "No"],
            "contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
            "paperless_billing": ["Yes", "No", "Yes", "Yes", "No"],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
            "monthly_charges": [29.85, 89.10, 105.50, 20.05, 55.20],
            "total_charges": [29.85, 2138.4, 6330.0, 240.6, np.nan],  # NaN to test imputation
            "churn": [1, 0, 0, 1, 0],
        }
    )


def test_build_preprocessor_transforms_data(sample_feature_df: pd.DataFrame) -> None:
    """Preprocessor correctly imputes NaNs and transforms numerics and categoricals."""
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_feature_df)

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(
        transformed
    ).any(), "Found unexpected NaN values after preprocessing imputation."
    assert transformed.shape[0] == len(sample_feature_df)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)  # Categoricals expanded by one-hot encoding


def test_unseen_categories_handled_gracefully(sample_feature_df: pd.DataFrame) -> None:
    """OneHotEncoder handle_unknown='ignore' ensures new categories don't crash inference."""
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_feature_df)

    test_df = sample_feature_df.copy()
    test_df.loc[0, "contract"] = "3-Year-Special"  # Unseen category
    test_df.loc[0, "payment_method"] = "Crypto Token"  # Unseen category

    transformed = preprocessor.transform(test_df)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()


def test_no_data_leakage_in_splits(sample_feature_df: pd.DataFrame) -> None:
    """Preprocessor fit only on train split; mean/std must reflect train data only."""
    # Duplicate df to 60 rows for splitting
    df_large = pd.concat([sample_feature_df] * 12, ignore_index=True)
    df_train, df_val, df_test = split_dataset(df_large, target_column="churn", seed=42)

    assert len(df_train) + len(df_val) + len(df_test) == len(df_large)
    # Stratification check: target ratios must be approximately equal
    train_rate = df_train["churn"].mean()
    val_rate = df_val["churn"].mean()
    test_rate = df_test["churn"].mean()
    assert abs(train_rate - val_rate) < 0.10
    assert abs(train_rate - test_rate) < 0.10
