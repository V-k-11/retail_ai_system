import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from src.utils.helpers import project_path


def load_model(model_uri: str | None = None):
    uri = model_uri or "models:/retail_model/latest"
    return mlflow.pyfunc.load_model(uri)


def predict_records(records: list[dict], model_uri: str | None = None) -> list:
    model = load_model(model_uri)
    predictions = model.predict(pd.DataFrame(records))
    return predictions.tolist()


def run() -> None:
    sample_path = project_path("data/predictions/sample_input.json")
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Missing sample input: {sample_path}. Create a JSON list of records first."
        )

    records = json.loads(sample_path.read_text())
    predictions = predict_records(records)

    output_path = project_path("data/predictions/latest_predictions.json")
    Path(output_path).write_text(json.dumps({"predictions": predictions}, indent=2))
    print(f"Wrote predictions -> {output_path}")

