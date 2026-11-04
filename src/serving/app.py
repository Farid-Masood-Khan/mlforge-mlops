"""FastAPI application for MLForge inference and model serving."""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.serving.predictor import ModelPredictor
from src.serving.schemas import (
    BatchPredictionResponse,
    CustomerChurnRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

DEFAULT_API_KEY = "mlforge-dev-secret-key-change-in-prod"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads model from MLflow registry at startup."""
    model_name = os.environ.get("MODEL_NAME", "mlforge-churn")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)

    try:
        predictor = ModelPredictor(
            model_name=model_name,
            alias=alias,
            tracking_uri=tracking_uri,
        )
        app.state.predictor = predictor
        print(f"[Serving] Successfully loaded model '{model_name}@{alias}' (v{predictor.version}).")
    except Exception as exc:
        app.state.predictor = None
        print(f"[Serving] WARNING: Failed to load model from registry: {exc}")

    yield


app = FastAPI(
    title="MLForge Churn Prediction API",
    description="Production-grade API serving churn predictions with schema validation and drift tracking.",
    version="0.1.0",
    lifespan=lifespan,
)


def verify_api_key(
    x_api_key: str | None = Header(default=None, description="API Access Key"),
) -> str:
    """Authenticates API requests via X-API-Key header using constant-time comparison."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-API-Key header.",
        )
    expected_key = os.environ.get("MLFORGE_API_KEY", DEFAULT_API_KEY)
    if not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided in X-API-Key header.",
        )
    return x_api_key


def get_predictor(request: Request) -> ModelPredictor:
    """Dependency injection for active ModelPredictor instance."""
    predictor: ModelPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable. Registry model failed to load.",
        )
    return predictor


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> JSONResponse:
    """Health check endpoint. Unauthenticated per TRD §7."""
    predictor: ModelPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "model_version": None,
                "details": "Model registry artifact not loaded.",
            },
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "model_loaded": True,
            "model_version": predictor.version,
            "details": f"Model '{predictor.model_name}' alias '{predictor.alias}' active.",
        },
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Model Info"],
)
def model_info(predictor: ModelPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Returns active model metadata, parameters, and holdout evaluation metrics."""
    return predictor.get_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference"],
)
def predict(
    payload: CustomerChurnRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Generates churn prediction for a single validated customer payload."""
    return predictor.predict_single(payload, source="api")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference"],
)
def predict_batch(
    payload: list[dict[str, Any]],
    predictor: ModelPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Batch inference with partial validation failure tolerance per BACKEND_SCHEMA §2."""
    if len(payload) > 1000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Batch payload exceeds 1,000 item ceiling.",
        )
    return predictor.predict_batch(payload, source="api")


@app.get(
    "/metrics",
    dependencies=[Depends(verify_api_key)],
    tags=["Monitoring"],
)
def metrics(predictor: ModelPredictor = Depends(get_predictor)) -> dict[str, Any]:
    """Returns runtime latency and request metrics in JSON format."""
    return predictor.get_metrics_summary()
