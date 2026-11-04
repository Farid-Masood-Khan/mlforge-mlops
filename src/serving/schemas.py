"""Pydantic data contracts for MLForge serving layer.

Mirrors Pandera training schemas strictly to prevent training/serving skew.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Literal definitions match Pandera categories exactly
GenderType = Literal["Female", "Male"]
PartnerType = Literal["Yes", "No"]
DependentsType = Literal["Yes", "No"]
PhoneServiceType = Literal["Yes", "No"]
MultipleLinesType = Literal["Yes", "No", "No phone service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
SubServiceType = Literal["Yes", "No", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaperlessBillingType = Literal["Yes", "No"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerChurnRequest(BaseModel):
    """Customer features payload for churn inference."""

    model_config = {"extra": "forbid"}

    gender: GenderType
    senior_citizen: int = Field(ge=0, le=1, description="0 for non-senior, 1 for senior")
    partner: PartnerType
    dependents: DependentsType
    tenure_months: int = Field(ge=0, le=72, description="Months customer has stayed with company")
    phone_service: PhoneServiceType
    multiple_lines: MultipleLinesType
    internet_service: InternetServiceType
    online_security: SubServiceType
    online_backup: SubServiceType
    device_protection: SubServiceType
    tech_support: SubServiceType
    streaming_tv: SubServiceType
    streaming_movies: SubServiceType
    contract: ContractType
    paperless_billing: PaperlessBillingType
    payment_method: PaymentMethodType
    monthly_charges: float = Field(ge=18.0, le=120.0, description="Monthly fee charged to customer")
    total_charges: float | None = Field(
        default=None,
        ge=0.0,
        description="Total charges over customer lifetime (imputed if null)",
    )


class ModelMeta(BaseModel):
    name: str
    version: str | int
    alias: str
    run_id: str


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: int = Field(ge=0, le=1)
    threshold: float
    model: ModelMeta
    request_id: str


class BatchPredictionItem(BaseModel):
    index: int
    churn_probability: float
    churn_prediction: int
    threshold: float


class BatchErrorItem(BaseModel):
    index: int
    error: str


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    errors: list[BatchErrorItem]
    total: int
    successful: int
    failed: int
    model: ModelMeta


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    name: str
    version: str | int
    alias: str
    run_id: str
    threshold: float
    trained_data_sha256: str
    holdout_metrics: dict[str, float]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_version: str | int | None = None
    details: str | None = None
