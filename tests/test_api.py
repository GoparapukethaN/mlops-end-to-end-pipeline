"""
Tests for the FastAPI churn prediction service.
"""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
      """Test health check endpoints."""

    def test_root(self, client):
              response = client.get("/")
              assert response.status_code == 200
              assert response.json()["status"] == "running"

    def test_health(self, client):
              response = client.get("/health")
              assert response.status_code == 200
              assert response.json()["status"] == "healthy"

    def test_liveness(self, client):
              response = client.get("/health/live")
              assert response.status_code == 200


class TestPredictionEndpoint:
      """Test prediction endpoint."""

    def test_predict_valid_input(self, client):
              payload = {
                            "tenure": 12,
                            "MonthlyCharges": 70.5,
                            "TotalCharges": 846.0,
                            "contract": "Month-to-month",
                            "PaymentMethod": "Electronic check"
              }
              response = client.post("/predict", json=payload)
              assert response.status_code in [200, 503]

    def test_predict_invalid_input(self, client):
              payload = {"tenure": -1}
              response = client.post("/predict", json=payload)
              assert response.status_code == 422


@pytest.fixture
def client():
      from src.api.main import app
      return TestClient(app)
