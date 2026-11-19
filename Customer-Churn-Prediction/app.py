"""FastAPI inference endpoint for the trained churn pipeline."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_pipeline.joblib"
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


@lru_cache
def load_model():
    # Only load artifacts trained by this project; joblib/pickle is unsafe for untrusted files.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing; run python train.py first")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(customer: Customer):
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    row = pd.DataFrame([customer.model_dump()])[artifact["features"]]
    probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
    return {"churn_probability": round(probability, 4),
            "prediction": "churn" if probability >= 0.5 else "stay",
            "threshold": 0.5}
