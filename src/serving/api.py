import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_URI = os.getenv("MODEL_URI", "models:/retail_model/latest")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = FastAPI(title="Retail AI System", version="0.1.0")
model = None
model_error = None


class PredictionRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1)


@app.on_event("startup")
def load_model() -> None:
    global model, model_error
    if DEMO_MODE:
        model = None
        model_error = "Demo mode enabled. MLflow model loading skipped."
        return

    try:
        import mlflow.pyfunc

        model = mlflow.pyfunc.load_model(MODEL_URI)
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)


@app.get("/")
def root() -> dict:
    return {
        "name": "Retail AI System",
        "status": "live",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "demo_mode": DEMO_MODE,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "model_error": model_error,
        "demo_mode": DEMO_MODE,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if model is None and DEMO_MODE:
        predictions = [_demo_reorder_prediction(record) for record in payload.records]
        return {
            "predictions": predictions,
            "mode": "demo",
            "message": "Demo heuristic used because no MLflow model is loaded.",
        }

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Train the model first or set MODEL_URI. Error: {model_error}",
        )
    predictions = model.predict(pd.DataFrame(payload.records))
    return {"predictions": predictions.tolist(), "mode": "mlflow"}


def _demo_reorder_prediction(record: dict) -> int:
    reorder_rate = float(record.get("product_reorder_rate", 0) or 0)
    user_orders = float(record.get("user_order_count", 0) or 0)
    days_since_prior = float(record.get("days_since_prior_order", 999) or 999)
    add_to_cart_order = float(record.get("add_to_cart_order", 999) or 999)

    likely_repeat_item = reorder_rate >= 0.5
    active_customer = user_orders >= 5 and days_since_prior <= 14
    high_intent_cart_position = add_to_cart_order <= 5
    return int((likely_repeat_item and active_customer) or high_intent_cart_position)
