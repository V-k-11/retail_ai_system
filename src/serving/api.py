import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_URI = os.getenv("MODEL_URI", "models:/retail_model/latest")

app = FastAPI(title="Retail AI System", version="0.1.0")
model = None


class PredictionRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1)


@app.on_event("startup")
def load_model() -> None:
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if model is None:
        raise RuntimeError("Model is not loaded")
    predictions = model.predict(pd.DataFrame(payload.records))
    return {"predictions": predictions.tolist()}

