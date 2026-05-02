from __future__ import annotations

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs() -> None:
    for relative in [
        "data/raw",
        "data/processed",
        "data/features",
        "data/predictions",
        "mlruns",
    ]:
        project_path(relative).mkdir(parents=True, exist_ok=True)


def use_pandas_engine() -> bool:
    return os.getenv("RETAIL_ENGINE", "").lower() == "pandas"


def get_spark(app_name: str) -> SparkSession:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
