"""Unit tests for the data validation gate (Pandera schemas).

Verifies that valid data passes and dirty/corrupted data fails loudly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import (
    DataValidationError,
    validate_dataframe,
)


@pytest.fixture
def valid_row() -> dict[str, object]:
    """Single valid row of raw Telco data."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "358.2",
        "Churn": "No",
    }


@pytest.fixture
def valid_df(valid_row: dict[str, object]) -> pd.DataFrame:
    """A valid DataFrame with 50 rows."""
    rows = []
    for i in range(50):
        row = valid_row.copy()
        row["customerID"] = f"CUST_{i:04d}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_fixture_passes(valid_df: pd.DataFrame) -> None:
    """Valid data passes and drops customerID."""
    result = validate_dataframe(valid_df, include_target=True)
    assert "customerID" not in result.columns
    assert "churn" in result.columns
    assert len(result) == 50
    assert result["churn"].isin([0, 1]).all()
    assert result["tenure_months"].dtype == "int64" or result["tenure_months"].dtype == "int32"


def test_missing_column_fails(valid_df: pd.DataFrame) -> None:
    """Dropping a required column fails loudly."""
    corrupted = valid_df.drop(columns=["Contract"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted)
    assert any("contract" in err.lower() for err in exc_info.value.errors)


def test_unseen_category_fails(valid_df: pd.DataFrame) -> None:
    """Injecting an unseen category raises validation error."""
    corrupted = valid_df.copy()
    corrupted.loc[0, "Contract"] = "Centennial Plan"
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted)
    assert any("contract" in err.lower() for err in exc_info.value.errors)


def test_out_of_range_numeric_fails(valid_df: pd.DataFrame) -> None:
    """Numeric out of defined range fails."""
    corrupted = valid_df.copy()
    corrupted.loc[0, "MonthlyCharges"] = 550.0  # Max is 120.0
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted)
    assert any("monthly_charges" in err.lower() for err in exc_info.value.errors)


def test_excess_nulls_fails(valid_df: pd.DataFrame) -> None:
    """Missing value rate > 5% fails validation."""
    corrupted = valid_df.copy()
    # 5 out of 50 is 10% nulls (> 5% ceiling)
    corrupted.loc[:5, "gender"] = None
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted)
    assert any("missing rate" in err.lower() for err in exc_info.value.errors)


def test_duplicate_customer_id_fails(valid_df: pd.DataFrame) -> None:
    """Duplicate customerID raises validation error."""
    corrupted = valid_df.copy()
    corrupted.loc[1, "customerID"] = corrupted.loc[0, "customerID"]
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted)
    assert any("duplicate" in err.lower() for err in exc_info.value.errors)


def test_total_charges_blank_coercion_and_ceiling(valid_df: pd.DataFrame) -> None:
    """Tenure=0 blank TotalCharges is coerced to NaN. If <= 2% it passes; > 2% fails."""
    # Test acceptable blank rate (1 out of 50 = 2%)
    acceptable = valid_df.copy()
    acceptable.loc[0, "tenure"] = 0
    acceptable.loc[0, "TotalCharges"] = " "
    result = validate_dataframe(acceptable)
    assert pd.isna(result.loc[0, "total_charges"])

    # Test excess blank rate (3 out of 50 = 6% > 2% ceiling)
    excess = valid_df.copy()
    excess.loc[:2, "TotalCharges"] = " "
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(excess)
    assert any("blank rate" in err.lower() for err in exc_info.value.errors)
