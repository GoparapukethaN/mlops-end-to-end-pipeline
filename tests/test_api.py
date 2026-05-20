"""Tests for the FastAPI churn prediction service."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def valid_prediction_payload() -> dict[str, object]:
    return {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "PaperlessBilling": "Yes",
    }


def test_root_reports_service_status(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "running",
        "service": "churn-prediction-api",
        "version": "1.0.0",
    }


def test_health_reports_model_state(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True


def test_readiness_requires_loaded_model(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_returns_churn_risk(client: TestClient) -> None:
    response = client.post("/predict", json=valid_prediction_payload())

    assert response.status_code == 200
    prediction = response.json()
    assert 0 <= prediction["churn_probability"] <= 1
    assert prediction["churn_prediction"] in (0, 1)
    assert prediction["risk_level"] in {"Low", "Medium", "High"}


def test_predict_rounds_probability_to_four_decimals(client: TestClient) -> None:
    response = client.post("/predict", json=valid_prediction_payload())

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.4744


def test_predict_rejects_incomplete_payload(client: TestClient) -> None:
    response = client.post("/predict", json={"tenure": -1})

    assert response.status_code == 422


def test_predict_rejects_negative_customer_values(client: TestClient) -> None:
    payload = valid_prediction_payload()
    payload["tenure"] = -1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_unknown_categorical_values(client: TestClient) -> None:
    payload = valid_prediction_payload()
    payload["Contract"] = "Lifetime"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_readiness_returns_unavailable_when_model_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.main.model_artifacts", None)

    response = client.get("/health/ready")

    assert response.status_code == 503


def test_predict_returns_unavailable_when_model_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.main.model_artifacts", None)

    response = client.post("/predict", json=valid_prediction_payload())

    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_metrics(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "predictions_total" in response.text
