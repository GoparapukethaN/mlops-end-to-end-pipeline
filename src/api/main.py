"""
FastAPI service for churn prediction with Prometheus metrics.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS = Counter("predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

# Global model variable
model_artifacts = None

YesNo = Literal["Yes", "No"]
ServiceYesNo = Literal["Yes", "No", "No internet service"]


class PredictionRequest(BaseModel):
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    gender: Literal["Female", "Male"] = "Male"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: ServiceYesNo = "No"
    OnlineBackup: ServiceYesNo = "No"
    DeviceProtection: ServiceYesNo = "No"
    TechSupport: ServiceYesNo = "No"
    StreamingTV: ServiceYesNo = "No"
    StreamingMovies: ServiceYesNo = "No"
    PaperlessBilling: YesNo = "Yes"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


def load_model():
    """Load the trained model."""
    global model_artifacts
    model_path = os.getenv("MODEL_PATH", "models/churn_model.joblib")
    try:
        model_artifacts = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_artifacts = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered API for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"status": "running", "service": "churn-prediction-api", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_artifacts is not None}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with PREDICTION_LATENCY.time():
        try:
            # Convert request to DataFrame
            data = request.model_dump()
            df = pd.DataFrame([data])

            # Get model components
            model = model_artifacts["model"]
            label_encoders = model_artifacts["label_encoders"]
            feature_columns = model_artifacts["feature_columns"]

            # Encode categorical features
            for col in df.select_dtypes(include=["object"]).columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    df[col] = (
                        df[col]
                        .astype(str)
                        .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                    )

            # Ensure columns match
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]

            # Make prediction
            proba = float(model.predict_proba(df)[0][1])
            prediction = int(proba >= 0.5)

            risk_level = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"

            PREDICTIONS.inc()

            return PredictionResponse(
                churn_probability=round(proba, 4),
                churn_prediction=prediction,
                risk_level=risk_level,
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
