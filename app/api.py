"""FastAPI inference service. Run: uvicorn app.api:app --reload."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.feature_engineering import add_features
from src.retention_engine import recommend_retention
from src.explainability import explain_prediction

MODEL_PATH = Path("models/churn_model.pkl")
app = FastAPI(title="ChurnGuard AI API", version="1.0.0")


class Customer(BaseModel):
    customer_id: str = "api-customer"
    signup_date: str = "2024-01-01"
    tenure_months: int = Field(..., ge=0)
    plan_type: str
    auto_pay: int = Field(0, ge=0, le=1)
    contract_type: str
    payment_method: str
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_tickets_30d: int = Field(0, ge=0)
    avg_resolution_time_hours: float = Field(8, ge=0)
    satisfaction_score: float = Field(3, ge=1, le=5)
    login_count_30d: int = Field(0, ge=0)
    session_minutes_30d: float = Field(0, ge=0)
    feature_usage_score: float = Field(50, ge=0, le=100)
    days_since_last_login: int = Field(0, ge=0)
    discount_applied: int = Field(0, ge=0, le=1)
    price_increase_last_3m: int = Field(0, ge=0, le=1)
    num_payment_failures_6m: int = Field(0, ge=0)
    referral_count: int = Field(0, ge=0)


def load_bundle():
    if not MODEL_PATH.exists():
        from src.train import train_model
        train_model(model_path=str(MODEL_PATH))
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(customer: Customer):
    try:
        raw = customer.model_dump()
        row = add_features(pd.DataFrame([raw]))
        bundle = load_bundle()
        probability = float(bundle["pipeline"].predict_proba(row[bundle["features"]])[0, 1])
        retention = recommend_retention(raw, probability)
        explanations = explain_prediction(bundle, row.iloc[0].to_dict())
        return {
            "churn_probability": round(probability, 4),
            "risk_level": retention["risk_level"],
            "top_reasons": explanations,
            "recommended_retention_offer": retention["actions"][0],
            "retention": retention,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
