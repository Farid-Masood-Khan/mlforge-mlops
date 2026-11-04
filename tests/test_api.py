"""Integration and schema parity tests for the FastAPI serving layer."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.data.validate import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_pandera_schema,
)
from src.serving.app import app
from src.serving.schemas import CustomerChurnRequest

VALID_HEADERS = {"X-API-Key": "mlforge-dev-secret-key-change-in-prod"}


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.5,
        "total_charges": 846.0,
    }


def test_schema_parity() -> None:
    """Schema Parity Test: asserts Pydantic request model mirrors Pandera schema exactly."""
    pandera_schema = get_pandera_schema(include_target=False)
    pandera_columns = set(pandera_schema.columns.keys())
    pydantic_fields = set(CustomerChurnRequest.model_fields.keys())

    # All feature columns in training schema must match API schema
    expected_features = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    assert (
        pandera_columns == expected_features
    ), "Pandera schema columns mismatch expected feature list"
    assert pydantic_fields == expected_features, "Pydantic fields mismatch expected feature list"


def test_api_unauthorized_without_key(valid_payload: dict[str, Any]) -> None:
    """Missing or invalid X-API-Key header returns 401 Unauthorized."""
    with TestClient(app) as client:
        # Missing key
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 401

        # Wrong key
        resp2 = client.post(
            "/predict",
            json=valid_payload,
            headers={"X-API-Key": "wrong-secret-token"},
        )
        assert resp2.status_code == 401


def test_health_endpoint() -> None:
    """GET /health is unauthenticated and returns 200 when model is loaded."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_version"] is not None


def test_predict_happy_path(valid_payload: dict[str, Any]) -> None:
    """POST /predict returns 200 with probability, prediction, and model metadata."""
    with TestClient(app) as client:
        resp = client.post("/predict", json=valid_payload, headers=VALID_HEADERS)
        assert resp.status_code == 200
        body = resp.json()

        assert "churn_probability" in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] in [0, 1]
        assert "model" in body
        assert body["model"]["name"] == "mlforge-churn"
        assert body["model"]["alias"] == "champion"
        assert "request_id" in body


def test_predict_validation_error_422(valid_payload: dict[str, Any]) -> None:
    """Invalid payload (e.g. unknown category) returns 422 Unprocessable Entity."""
    corrupted = valid_payload.copy()
    corrupted["contract"] = "Centennial Plan"  # Unseen enum

    with TestClient(app) as client:
        resp = client.post("/predict", json=corrupted, headers=VALID_HEADERS)
        assert resp.status_code == 422


def test_model_info_endpoint() -> None:
    """GET /model/info returns registered model version, alias, and holdout metrics."""
    with TestClient(app) as client:
        resp = client.get("/model/info", headers=VALID_HEADERS)
        assert resp.status_code == 200
        info = resp.json()

        assert info["alias"] == "champion"
        assert "holdout_metrics" in info
        assert "pr_auc" in info["holdout_metrics"]


def test_batch_prediction_with_partial_errors(valid_payload: dict[str, Any]) -> None:
    """POST /predict/batch handles valid rows and reports row-level errors for invalid ones."""
    bad_payload = valid_payload.copy()
    bad_payload["monthly_charges"] = 550.0  # Invalid > 120

    batch_payload = [valid_payload, bad_payload, valid_payload]

    with TestClient(app) as client:
        resp = client.post("/predict/batch", json=batch_payload, headers=VALID_HEADERS)
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"] == 3
        assert body["successful"] == 2
        assert body["failed"] == 1
        assert len(body["predictions"]) == 2
        assert len(body["errors"]) == 1
        assert body["errors"][0]["index"] == 1


def test_metrics_endpoint() -> None:
    """GET /metrics returns server metrics JSON."""
    with TestClient(app) as client:
        resp = client.get("/metrics", headers=VALID_HEADERS)
        assert resp.status_code == 200
        metrics = resp.json()
        assert "requests_total" in metrics
        assert "latency_p95_ms" in metrics
