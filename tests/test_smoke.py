"""Smoke tests for MLForge Phase 0 setup."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Milestone M0/M3: /health returns 200 and valid status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_config_exists_and_valid() -> None:
    """Assert configs/train_config.yaml is present and contains required sections."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"
    assert config_path.exists(), f"Configuration file not found at {config_path}"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "project" in config
    assert "data" in config
    assert "model" in config
    assert "promotion" in config
    assert "drift" in config
    assert config["project"]["name"] == "mlforge-churn"
