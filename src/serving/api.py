import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field


MODEL_URI = os.getenv("MODEL_URI", "models:/retail_model/latest")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
USE_MLFLOW_MODEL = os.getenv("USE_MLFLOW_MODEL", "false").lower() == "true"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "data/predictions/retail_model.joblib")
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4")

app = FastAPI(title="Retail AI System", version="0.1.0")
model = None
model_error = None
model_source = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        errors.append({"field": field, "message": error.get("msg")})
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": errors})


class RetailOrderFeatures(BaseModel):
    order_number: int = Field(..., example=3)
    order_dow: int = Field(..., ge=0, le=6, example=1)
    order_hour_of_day: int = Field(..., ge=0, le=23, example=10)
    days_since_prior_order: float = Field(..., ge=0, example=7)
    add_to_cart_order: int = Field(..., ge=1, example=1)
    user_order_count: int = Field(..., ge=0, example=5)
    product_order_count: int = Field(..., ge=0, example=100)
    product_reorder_rate: float = Field(..., ge=0, le=1, example=0.6)
    aisle_id: int = Field(..., ge=1, example=24)
    department_id: int = Field(..., ge=1, example=4)


class PredictionRequest(BaseModel):
    records: list[RetailOrderFeatures] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    prediction: int
    label: str
    confidence_hint: str
    reason: str


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    mode: str
    message: str | None = None


class AssistantRequest(BaseModel):
    question: str = Field(
        ...,
        example="Why did the model predict this product will be reordered?",
    )
    context: RetailOrderFeatures | None = None


class AssistantResponse(BaseModel):
    answer: str
    relevant_system_parts: list[str]
    genai_role: str


@app.on_event("startup")
def load_model() -> None:
    global model, model_error, model_source
    if DEMO_MODE:
        model = None
        model_error = "Demo mode enabled. MLflow model loading skipped."
        model_source = "demo"
        return

    if not USE_MLFLOW_MODEL:
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            model_error = None
            model_source = "local_joblib"
            return
        except Exception as local_exc:
            model = None
            model_source = None
            model_error = f"Local model load failed from {LOCAL_MODEL_PATH}: {local_exc}"
            return

    try:
        import mlflow.pyfunc

        model = mlflow.pyfunc.load_model(MODEL_URI)
        model_error = None
        model_source = "mlflow"
    except Exception as exc:
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            model_error = None
            model_source = "local_joblib"
        except Exception as local_exc:
            model = None
            model_source = None
            model_error = (
                f"MLflow load failed: {exc}. Local model load failed from "
                f"{LOCAL_MODEL_PATH}: {local_exc}"
            )


@app.get("/")
def root() -> dict:
    return {
        "name": "Retail AI System",
        "status": "live",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "assistant": "/assistant",
        "demo_mode": DEMO_MODE,
        "purpose": "Predict whether a customer is likely to reorder a product and explain the result.",
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retail AI System Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }
    main { max-width: 980px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 6px; font-size: 30px; }
    p { margin: 0 0 20px; color: #53616f; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    label { display: grid; gap: 6px; font-size: 14px; font-weight: 700; }
    input, textarea { padding: 10px 12px; border: 1px solid #c9d1d9; border-radius: 6px; font-size: 15px; }
    textarea { min-height: 78px; resize: vertical; }
    .actions { display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font-weight: 700; cursor: pointer; }
    .primary { background: #155eef; color: white; }
    .secondary { background: #17202a; color: white; }
    .panel { background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 18px; margin-top: 18px; }
    .result { white-space: pre-wrap; font-family: Consolas, monospace; font-size: 14px; }
    .error { color: #b42318; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } main { padding: 18px; } }
  </style>
</head>
<body>
<main>
  <h1>Retail AI System</h1>
  <p>Input customer and product behavior, then get a reorder prediction and business explanation.</p>

  <section class="panel">
    <div class="grid">
      <label>Order number <input id="order_number" type="number" value="10"></label>
      <label>Order day of week <input id="order_dow" type="number" min="0" max="6" value="1"></label>
      <label>Order hour <input id="order_hour_of_day" type="number" min="0" max="23" value="5"></label>
      <label>Days since prior order <input id="days_since_prior_order" type="number" value="7"></label>
      <label>Add to cart order <input id="add_to_cart_order" type="number" value="1"></label>
      <label>User order count <input id="user_order_count" type="number" value="5"></label>
      <label>Product order count <input id="product_order_count" type="number" value="100"></label>
      <label>Product reorder rate <input id="product_reorder_rate" type="number" step="0.01" min="0" max="1" value="0.6"></label>
      <label>Aisle ID <input id="aisle_id" type="number" value="24"></label>
      <label>Department ID <input id="department_id" type="number" value="4"></label>
    </div>
    <div class="actions">
      <button class="primary" onclick="predict()">Predict Reorder</button>
      <button class="secondary" onclick="explainPrediction()">Explain With Assistant</button>
    </div>
  </section>

  <section class="panel">
    <strong>Output</strong>
    <div id="output" class="result">Run a prediction to see the result.</div>
  </section>
</main>
<script>
function record() {
  const numberFields = [
    "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order",
    "add_to_cart_order", "user_order_count", "product_order_count",
    "product_reorder_rate", "aisle_id", "department_id"
  ];
  const data = {};
  for (const field of numberFields) data[field] = Number(document.getElementById(field).value);
  return data;
}

function show(value, isError=false) {
  const output = document.getElementById("output");
  output.className = isError ? "result error" : "result";
  output.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function callApi(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw data;
  return data;
}

async function predict() {
  try {
    const data = await callApi("/predict", {records: [record()]});
    const first = data.predictions[0];
    show({
      result: first.label,
      prediction_value: first.prediction,
      reason: first.reason,
      model_mode: data.mode
    });
  } catch (error) {
    show(error, true);
  }
}

async function explainPrediction() {
  try {
    const data = await callApi("/assistant", {
      question: "Explain this reorder prediction in simple business terms",
      context: record()
    });
    show(data.answer);
  } catch (error) {
    show(error, true);
  }
}
</script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "model_error": model_error,
        "model_source": model_source,
        "demo_mode": DEMO_MODE,
        "local_model_path": LOCAL_MODEL_PATH,
        "grok_enabled": bool(XAI_API_KEY),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    records = [record.model_dump() for record in payload.records]

    if model is None and DEMO_MODE:
        predictions = [_demo_reorder_prediction(record) for record in records]
        return PredictionResponse(
            predictions=predictions,
            mode="demo",
            message="Demo heuristic used because no MLflow model is loaded.",
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Train the model first or set MODEL_URI. Error: {model_error}",
        )
    model_predictions = model.predict(pd.DataFrame(records))
    predictions = [
        PredictionResult(
            prediction=int(value),
            label="likely_reordered" if int(value) == 1 else "not_likely_reordered",
            confidence_hint="model_score",
            reason="Generated by the MLflow model registered for retail reorder prediction.",
        )
        for value in model_predictions.tolist()
    ]
    return PredictionResponse(predictions=predictions, mode=model_source or "model")


@app.post("/assistant", response_model=AssistantResponse)
def assistant(payload: AssistantRequest) -> AssistantResponse:
    """Demo GenAI assistant layer for explaining the ML system."""
    context = payload.context.model_dump() if payload.context else None
    if context:
        prediction = _demo_reorder_prediction(context)
        fallback_answer = _build_demo_assistant_answer(payload.question, context, prediction)
    else:
        fallback_answer = (
            "This project predicts Instacart-style product reorders using a modular ML "
            "pipeline. GenAI is useful as an assistant layer: it can explain predictions, "
            "summarize monitoring results, answer questions about features, and help a "
            "business user understand why retraining may be needed."
        )

    answer = _ask_grok(payload.question, context, fallback_answer)
    return AssistantResponse(
        answer=answer,
        relevant_system_parts=[
            "FastAPI serving API",
            "PySpark feature engineering",
            "MLflow model lifecycle",
            "Drift monitoring",
            "RAG assistant concept",
        ],
        genai_role="Explanation and retrieval assistant, not the core reorder prediction model.",
    )


def _demo_reorder_prediction(record: dict) -> PredictionResult:
    reorder_rate = float(record.get("product_reorder_rate", 0) or 0)
    user_orders = float(record.get("user_order_count", 0) or 0)
    days_since_prior = float(record.get("days_since_prior_order", 999) or 999)
    add_to_cart_order = float(record.get("add_to_cart_order", 999) or 999)

    likely_repeat_item = reorder_rate >= 0.5
    active_customer = user_orders >= 5 and days_since_prior <= 14
    high_intent_cart_position = add_to_cart_order <= 5
    prediction = int((likely_repeat_item and active_customer) or high_intent_cart_position)

    reasons = []
    if likely_repeat_item:
        reasons.append("product has a strong historical reorder rate")
    if active_customer:
        reasons.append("customer has enough order history and recent activity")
    if high_intent_cart_position:
        reasons.append("product appears early in the cart")
    if not reasons:
        reasons.append("signals are not strong enough for a reorder")

    return PredictionResult(
        prediction=prediction,
        label="likely_reordered" if prediction == 1 else "not_likely_reordered",
        confidence_hint="high" if prediction == 1 and len(reasons) >= 2 else "medium",
        reason=", ".join(reasons),
    )


def _build_demo_assistant_answer(question: str, context: dict, prediction: PredictionResult) -> str:
    return (
        f"Question: {question}\n"
        f"Prediction: {prediction.label} ({prediction.prediction}).\n"
        f"Reason: {prediction.reason}.\n"
        "Business interpretation: this helps a retailer decide which products to recommend, "
        "prioritize, or monitor for reorder behavior. In a production version, this answer "
        "can be generated with RAG over feature definitions, MLflow metrics, model cards, "
        "and monitoring reports."
    )


def _ask_grok(question: str, context: dict | None, fallback_answer: str) -> str:
    if not XAI_API_KEY:
        return fallback_answer

    try:
        import httpx

        prompt = (
            "You are a retail ML assistant. Explain predictions from an Instacart-style "
            "reorder model in simple business language. Be concise.\n\n"
            f"Question: {question}\n"
            f"Prediction context: {context}\n"
            f"Fallback system explanation: {fallback_answer}"
        )
        response = httpx.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": XAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You explain retail ML model outputs for business users.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"{fallback_answer}\n\nGrok API fallback used because the API call failed: {exc}"
