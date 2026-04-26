# Retail AI System

Declarative, modular, self-improving ML pipeline for the open-source Instacart dataset. The project runs locally in VS Code and avoids any Databricks dependency.

## Architecture

The pipeline is controlled by YAML configs in `config/`:

- `pipeline.yaml` decides which stages run and in what order.
- `features.yaml` defines feature and label columns.
- `model.yaml` defines model type and parameters.
- `training.yaml` defines split, MLflow, and drift thresholds.

Core stages:

1. `ingestion`: reads Instacart CSV files from `data/raw`.
2. `processing`: joins orders, order products, and products.
3. `feature_engineering`: builds user and product features.
4. `training`: trains a scikit-learn model and logs it to MLflow.
5. `evaluation`: checks metrics against the drift threshold.
6. `deployment`: optional batch prediction step.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the Instacart CSV files and place them in `data/raw/`:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

## Run The Pipeline

```bash
python main.py
```

Run only when drift is detected:

```bash
python main.py --check-drift
```

## Serve The Model

```bash
uvicorn src.serving.api:app --reload
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"records\":[{\"order_number\":3,\"order_dow\":1,\"order_hour_of_day\":10,\"days_since_prior_order\":7,\"add_to_cart_order\":1,\"user_order_count\":5,\"product_order_count\":100,\"product_reorder_rate\":0.6,\"aisle_id\":24,\"department_id\":4}]}"
```

## RAG Assistant

`src/rag/rag_pipeline.py` contains a minimal FAISS-backed retrieval helper. Set `OPENAI_API_KEY` before using it with `OpenAIEmbeddings`.

## Tests

```bash
pytest
```

