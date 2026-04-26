from importlib import import_module
from typing import Callable

from src.utils.helpers import load_yaml, project_path


STEP_REGISTRY = {
    "ingestion": "src.ingestion.ingest:run",
    "processing": "src.processing.transform:run",
    "feature_engineering": "src.features.build_features:run",
    "training": "src.models.train:run",
    "evaluation": "src.models.evaluate:run",
    "deployment": "src.models.predict:run",
}


def _load_callable(path: str) -> Callable[[], None]:
    module_name, function_name = path.split(":")
    module = import_module(module_name)
    return getattr(module, function_name)


def run_pipeline(config_path: str = "config/pipeline.yaml") -> None:
    config = load_yaml(project_path(config_path))

    for step in config["steps"]:
        if step not in STEP_REGISTRY:
            raise ValueError(f"Unknown pipeline step: {step}")

        print(f"Running step: {step}")
        _load_callable(STEP_REGISTRY[step])()

