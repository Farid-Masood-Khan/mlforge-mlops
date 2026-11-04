"""Inference wrapper, MLflow model loading, structured prediction logging, and metrics."""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.serving.schemas import (
    BatchErrorItem,
    BatchPredictionItem,
    BatchPredictionResponse,
    CustomerChurnRequest,
    ModelInfoResponse,
    ModelMeta,
    PredictionResponse,
)


class ModelPredictor:
    """Manages loaded pipeline, handles inference, structured JSONL logs, and runtime metrics."""

    def __init__(
        self,
        model_name: str = "mlforge-churn",
        alias: str = "champion",
        tracking_uri: str | None = None,
        logs_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        tracking_env = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.tracking_uri: str = tracking_uri if tracking_uri is not None else tracking_env
        self.logs_dir = logs_dir or Path(__file__).resolve().parent.parent.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_log_file = self.logs_dir / "predictions.jsonl"

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Load model and lineage metadata
        self.pipeline: Pipeline
        self.version: str
        self.run_id: str
        self.threshold: float
        self.data_sha256: str
        self.holdout_metrics: dict[str, float]

        self._load_from_registry()

        # Runtime performance metrics
        self.requests_total: int = 0
        self.predictions_total: int = 0
        self.errors_total: int = 0
        self.latencies_ms: list[float] = []

    def _load_from_registry(self) -> None:
        """Loads model pipeline and run parameters from MLflow model registry."""
        model_uri = f"models:/{self.model_name}@{self.alias}"
        self.pipeline = mlflow.sklearn.load_model(model_uri)

        model_version_obj = self.client.get_model_version_by_alias(self.model_name, self.alias)
        self.version = str(model_version_obj.version)
        self.run_id = str(model_version_obj.run_id)

        # Retrieve threshold and metrics from the run
        run = self.client.get_run(self.run_id)
        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags

        self.threshold = float(params.get("optimal_threshold", 0.50))
        self.data_sha256 = str(tags.get("data_sha256", "unknown"))
        self.holdout_metrics = {
            "roc_auc": float(metrics.get("roc_auc", 0.0)),
            "pr_auc": float(metrics.get("pr_auc", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "log_loss": float(metrics.get("log_loss", 0.0)),
        }

    def predict_single(
        self,
        request: CustomerChurnRequest,
        source: str = "api",
        request_id: str | None = None,
    ) -> PredictionResponse:
        """Predicts churn probability for a single customer payload."""
        start_time = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())
        self.requests_total += 1

        try:
            input_dict = request.model_dump()
            df = pd.DataFrame([input_dict])

            y_proba = float(self.pipeline.predict_proba(df)[0, 1])
            y_pred = int(y_proba >= self.threshold)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.latencies_ms.append(elapsed_ms)
            if len(self.latencies_ms) > 1000:
                self.latencies_ms = self.latencies_ms[-1000:]
            self.predictions_total += 1

            meta = ModelMeta(
                name=self.model_name,
                version=self.version,
                alias=self.alias,
                run_id=self.run_id,
            )

            # Log prediction JSONL per TRD §4
            self._log_prediction(
                req_id=req_id,
                features=input_dict,
                probability=y_proba,
                prediction=y_pred,
                source=source,
            )

            return PredictionResponse(
                churn_probability=round(y_proba, 4),
                churn_prediction=y_pred,
                threshold=self.threshold,
                model=meta,
                request_id=req_id,
            )

        except Exception:
            self.errors_total += 1
            raise

    def predict_batch(
        self,
        items: list[dict[str, Any]],
        source: str = "api",
    ) -> BatchPredictionResponse:
        """Runs batch predictions with partial error tolerance per BACKEND_SCHEMA §2."""
        self.requests_total += 1
        valid_rows: list[dict[str, Any]] = []
        valid_indices: list[int] = []
        errors: list[BatchErrorItem] = []

        # Validate rows individually
        for idx, item in enumerate(items):
            try:
                validated_req = CustomerChurnRequest.model_validate(item)
                valid_rows.append(validated_req.model_dump())
                valid_indices.append(idx)
            except Exception as e:
                errors.append(BatchErrorItem(index=idx, error=str(e)))

        predictions: list[BatchPredictionItem] = []
        if valid_rows:
            df = pd.DataFrame(valid_rows)
            probas = self.pipeline.predict_proba(df)[:, 1]
            for orig_idx, proba in zip(valid_indices, probas, strict=False):
                prob_val = float(proba)
                pred_val = int(prob_val >= self.threshold)
                predictions.append(
                    BatchPredictionItem(
                        index=orig_idx,
                        churn_probability=round(prob_val, 4),
                        churn_prediction=pred_val,
                        threshold=self.threshold,
                    )
                )
                self.predictions_total += 1

                # Log each record to JSONL
                self._log_prediction(
                    req_id=str(uuid.uuid4()),
                    features=valid_rows[len(predictions) - 1],
                    probability=prob_val,
                    prediction=pred_val,
                    source=source,
                )

        if errors:
            self.errors_total += len(errors)

        meta = ModelMeta(
            name=self.model_name,
            version=self.version,
            alias=self.alias,
            run_id=self.run_id,
        )

        return BatchPredictionResponse(
            predictions=predictions,
            errors=errors,
            total=len(items),
            successful=len(predictions),
            failed=len(errors),
            model=meta,
        )

    def _log_prediction(
        self,
        req_id: str,
        features: dict[str, Any],
        probability: float,
        prediction: int,
        source: str,
    ) -> None:
        """Appends one prediction record to logs/predictions.jsonl."""
        record = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "request_id": req_id,
            "source": source,
            "model_version": self.version,
            "probability": round(probability, 4),
            "prediction": prediction,
            "features": features,
        }
        with open(self.prediction_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_info(self) -> ModelInfoResponse:
        """Returns model registry and performance metadata."""
        return ModelInfoResponse(
            name=self.model_name,
            version=self.version,
            alias=self.alias,
            run_id=self.run_id,
            threshold=self.threshold,
            trained_data_sha256=self.data_sha256,
            holdout_metrics=self.holdout_metrics,
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Returns runtime server metrics for /metrics endpoint."""
        p95 = float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0
        p50 = float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0
        return {
            "requests_total": self.requests_total,
            "predictions_total": self.predictions_total,
            "errors_total": self.errors_total,
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "model_version": self.version,
            "model_alias": self.alias,
        }
