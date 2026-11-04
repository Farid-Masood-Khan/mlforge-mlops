"""Data validation gate for MLForge using Pandera schemas.

Enforces data contracts, types, allowed categories, missingness ceilings,
and dataset quirks (e.g., tenure=0 TotalCharges blanks).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
import yaml
from pandera import Check, Column, DataFrameSchema

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "train_config.yaml"
)

# Canonical column mapping from raw Telco format to standard snake_case schema
RAW_TO_CANONICAL_COLUMNS = {
    "customerID": "customerID",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "senior_citizen": "senior_citizen",
    "Partner": "partner",
    "partner": "partner",
    "Dependents": "dependents",
    "dependents": "dependents",
    "tenure": "tenure_months",
    "tenure_months": "tenure_months",
    "PhoneService": "phone_service",
    "phone_service": "phone_service",
    "MultipleLines": "multiple_lines",
    "multiple_lines": "multiple_lines",
    "InternetService": "internet_service",
    "internet_service": "internet_service",
    "OnlineSecurity": "online_security",
    "online_security": "online_security",
    "OnlineBackup": "online_backup",
    "online_backup": "online_backup",
    "DeviceProtection": "device_protection",
    "device_protection": "device_protection",
    "TechSupport": "tech_support",
    "tech_support": "tech_support",
    "StreamingTV": "streaming_tv",
    "streaming_tv": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "streaming_movies": "streaming_movies",
    "Contract": "contract",
    "contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "paperless_billing": "paperless_billing",
    "PaymentMethod": "payment_method",
    "payment_method": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "monthly_charges": "monthly_charges",
    "TotalCharges": "total_charges",
    "total_charges": "total_charges",
    "Churn": "churn",
    "churn": "churn",
}

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
]


class DataValidationError(Exception):
    """Raised when incoming dataset violates data contracts or schema validation rules."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


def get_pandera_schema(include_target: bool = True) -> DataFrameSchema:
    """Builds the strict Pandera schema per BACKEND_SCHEMA §1.1."""
    columns: dict[str, Column] = {
        "gender": Column(
            pa.String,
            Check.isin(["Female", "Male"]),
            nullable=False,
            coerce=True,
        ),
        "senior_citizen": Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
        "partner": Column(
            pa.String,
            Check.isin(["Yes", "No"]),
            nullable=False,
            coerce=True,
        ),
        "dependents": Column(
            pa.String,
            Check.isin(["Yes", "No"]),
            nullable=False,
            coerce=True,
        ),
        "tenure_months": Column(
            pa.Int,
            Check.in_range(0, 72),
            nullable=False,
            coerce=True,
        ),
        "phone_service": Column(
            pa.String,
            Check.isin(["Yes", "No"]),
            nullable=False,
            coerce=True,
        ),
        "multiple_lines": Column(
            pa.String,
            Check.isin(["Yes", "No", "No phone service"]),
            nullable=False,
            coerce=True,
        ),
        "internet_service": Column(
            pa.String,
            Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
            coerce=True,
        ),
        "online_security": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "online_backup": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "device_protection": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "tech_support": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "streaming_tv": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "streaming_movies": Column(
            pa.String,
            Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            coerce=True,
        ),
        "contract": Column(
            pa.String,
            Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
            coerce=True,
        ),
        "paperless_billing": Column(
            pa.String,
            Check.isin(["Yes", "No"]),
            nullable=False,
            coerce=True,
        ),
        "payment_method": Column(
            pa.String,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            nullable=False,
            coerce=True,
        ),
        "monthly_charges": Column(
            pa.Float,
            Check.in_range(18.0, 120.0),
            nullable=False,
            coerce=True,
        ),
        "total_charges": Column(
            pa.Float,
            Check.greater_than_or_equal_to(0.0),
            nullable=True,  # Imputed during feature pipeline
            coerce=True,
        ),
    }

    if include_target:
        columns["churn"] = Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        )

    return DataFrameSchema(columns=columns, strict=False, coerce=True)


def validate_dataframe(
    df_raw: pd.DataFrame,
    include_target: bool = True,
    max_missing_feature_pct: float = 0.05,
    max_raw_blank_total_charges_pct: float = 0.02,
) -> pd.DataFrame:
    """Validates raw DataFrame against strict schemas and domain integrity rules.

    Returns the validated, type-coerced DataFrame with customerID removed.
    Raises DataValidationError on contract violations.
    """
    errors: list[str] = []
    df = df_raw.copy()

    # 1. Normalize column names
    rename_dict = {}
    for col in df.columns:
        if col in RAW_TO_CANONICAL_COLUMNS:
            rename_dict[col] = RAW_TO_CANONICAL_COLUMNS[col]
    df.rename(columns=rename_dict, inplace=True)

    # 2. Check customerID if present
    if "customerID" in df.columns:
        if df["customerID"].isnull().any():
            errors.append("customerID contains null values.")
        if df["customerID"].duplicated().any():
            errors.append(f"customerID contains {df['customerID'].duplicated().sum()} duplicates.")
        # Drop customerID per BACKEND_SCHEMA §1.1
        df = df.drop(columns=["customerID"])

    # 3. Check for missing required columns
    expected_features = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    if include_target:
        expected_features.add("churn")
    missing_cols = expected_features - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        raise DataValidationError("Schema contract violation: missing columns", errors)

    # 4. Handle total_charges coercion and raw blank rate check
    if "total_charges" in df.columns:
        raw_tc = df["total_charges"].astype(str).str.strip()
        blank_mask = (raw_tc == "") | (raw_tc.str.lower() == "nan") | (raw_tc.str.lower() == "null")
        blank_rate = float(blank_mask.mean())

        if blank_rate > max_raw_blank_total_charges_pct:
            errors.append(
                f"total_charges blank rate ({blank_rate:.2%}) exceeds ceiling ({max_raw_blank_total_charges_pct:.2%})."
            )

        # Coerce to float
        df["total_charges"] = pd.to_numeric(raw_tc, errors="coerce")

    # 5. Encode target if present
    if include_target and "churn" in df.columns:
        if df["churn"].dtype == object or df["churn"].dtype.name == "category":
            val_map = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
            churn_series = df["churn"].astype(str).str.strip().str.lower()
            unrecognized = set(churn_series) - set(val_map.keys())
            if unrecognized:
                errors.append(f"Unrecognized values in churn target: {unrecognized}")
            else:
                df["churn"] = churn_series.map(val_map)

    # 6. Check missing value ceilings (>5% fails validation)
    for col in expected_features:
        if col in df.columns and col != "total_charges":
            missing_rate = float(df[col].isnull().mean())
            if missing_rate > max_missing_feature_pct:
                errors.append(
                    f"Feature '{col}' missing rate ({missing_rate:.2%}) exceeds ceiling ({max_missing_feature_pct:.2%})."
                )

    # 7. Pandera schema execution
    schema = get_pandera_schema(include_target=include_target)
    try:
        validated_df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as schema_err:
        for failure in schema_err.failure_cases.itertuples():
            col = getattr(failure, "column", "unknown")
            check = getattr(failure, "check", "unknown")
            case = getattr(failure, "failure_case", "unknown")
            errors.append(f"Schema violation on '{col}' (check: {check}, case: {case})")
    except Exception as exc:
        errors.append(f"Schema validation error: {exc}")

    if errors:
        raise DataValidationError(f"Validation failed with {len(errors)} error(s)", errors)

    return validated_df


def run_validation(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Reads raw CSV, validates against Pandera schema, and saves validated dataset."""
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    base_dir = config_path.parent.parent
    raw_path: Path = base_dir / str(config["data"]["raw_dir"]) / str(config["data"]["raw_filename"])
    processed_dir: Path = base_dir / str(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = processed_dir / "validated_churn.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run 'download.py' first.")

    print(f"[Validate] Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)

    val_config = config["data"]["validation"]
    validated_df = validate_dataframe(
        df_raw,
        include_target=True,
        max_missing_feature_pct=val_config["max_missing_feature_pct"],
        max_raw_blank_total_charges_pct=val_config["max_raw_blank_total_charges_pct"],
    )

    validated_df.to_parquet(output_path, index=False)
    print(f"[Validate] Validation passed! Saved {len(validated_df)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate raw dataset against schema contracts.")
    args = parser.parse_args()
    try:
        run_validation()
    except DataValidationError as err:
        print(f"[Validate] FATAL: {err}")
        for e in err.errors:
            print(f"  - {e}")
        exit(1)
