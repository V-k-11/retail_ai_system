import json
import os

from src.monitoring.drift import check_drift
from src.utils.helpers import load_yaml, project_path


def run() -> None:
    metrics_path = project_path("data/predictions/latest_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    training_config = load_yaml(
        project_path(os.getenv("RETAIL_TRAINING_CONFIG", "config/training.yaml"))
    )["training"]

    needs_retraining = check_drift(
        metrics,
        metric_name=training_config["target_metric"],
        threshold=training_config["min_metric_threshold"],
    )

    status = "retraining recommended" if needs_retraining else "model stable"
    print(f"Evaluation complete: {metrics}. Status: {status}.")
