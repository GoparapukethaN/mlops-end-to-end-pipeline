"""
FastAPI application for churn prediction service.
"""
import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.models.train import ChurnModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CHURN_PREDICTIONS = Counter('churn_predictions', 'Predictions by outcome', ['outcome'])

app = FastAPI(
      title="Churn Prediction API",
      description="API for predicting customer churn",
      version="1.0.0"
)

app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
)

# Load model on startup
model = ChurnModel()
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")


class CustomerData(BaseModel):
      """Input schema for prediction."""
      tenure: int = Field(..., ge=0, description="Months with company")
      monthly_charges: float = Field(..., ge=0, alias="MonthlyCharges")
      total_charges: float = Field(..., ge=0, alias="TotalCharges")
      contract: str = Field(..., description="Contract type")
      payment_method: str = Field(..., alias="PaymentMethod")
      gender: Optional[str] = "Male"
      senior_citizen: Optional[int] = Field(0, alias="SeniorCitizen")
      partner: Optional[str] = "No"
      dependents: Optional[str] = "No"
      phone_service: Optional[str] = Field("Yes", alias="PhoneService")
      internet_service: Optional[str] = Field("Fiber optic", alias="InternetService")

    class Config:
              populate_by_name = True


class PredictionResponse(BaseModel):
      """Output schema for prediction."""
      prediction: str
      probability: float
      model_version: str = "1.0.0"


@app.on_event("startup")
async def load_model():
      """Load model on startup."""
      try:
                model.load(MODEL_PATH)
                logger.info("Model loaded successfully")
except Exception as e:
        logger.warning(f"Could not load model: {e}. Will need to train first.")


@app.get("/")
async def root():
      """Root endpoint."""
      return {"message": "Churn Prediction API", "status": "running"}


@app.get("/health")
async def health():
      """Health check endpoint."""
      return {"status": "healthy"}


@app.get("/health/ready")
async def readiness():
      """Readiness probe."""
      if model.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return {"status": "ready"}


@app.get("/health/live")
async def liveness():
      """Liveness probe."""
    return {"status": "alive"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
      """Make churn prediction."""
    start_time = time.time()

    try:
              # Convert to dict for model
              import pandas as pd
              data = pd.DataFrame([customer.model_dump(by_alias=True)])

        # Get prediction
              proba = model.predict_proba(data)[0]
              prediction = "churn" if proba >= 0.5 else "no_churn"

        # Update metrics
              PREDICTIONS.inc()
              CHURN_PREDICTIONS.labels(outcome=prediction).inc()
              PREDICTION_LATENCY.observe(time.time() - start_time)

        return PredictionResponse(
                      prediction=prediction,
                      probability=round(float(proba), 3)
        )
except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
      """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
      import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
