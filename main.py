import argparse
import json
import os

from src.monitoring.drift import check_drift
from src.pipelines.pipeline_runner import run_pipeline
from src.utils.helpers import load_yaml, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Declarative Retail AI pipeline runner")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Only run the pipeline if latest metrics indicate drift",
    )
    args = parser.parse_args()

    if not args.check_drift:
        run_pipeline(args.config)
        return

    metrics_path = project_path("data/predictions/latest_metrics.json")
    if not metrics_path.exists():
        print("No metrics found. Running full pipeline.")
        run_pipeline(args.config)
        return

    metrics = json.loads(metrics_path.read_text())
    training_config = load_yaml(
        project_path(os.getenv("RETAIL_TRAINING_CONFIG", "config/training.yaml"))
    )["training"]
    if check_drift(
        metrics,
        metric_name=training_config["target_metric"],
        threshold=training_config["min_metric_threshold"],
    ):
        print("Drift detected. Retraining...")
        run_pipeline(args.config)
    else:
        print("Model stable. No retraining needed.")


if __name__ == "__main__":
    main()
