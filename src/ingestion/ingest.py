from pathlib import Path
import os

import pandas as pd

from src.utils.helpers import ensure_dirs, get_spark, project_path, use_pandas_engine


RAW_DIR = project_path(os.getenv("RETAIL_RAW_DIR", "data/raw"))
PROCESSED_DIR = project_path("data/processed")
RAW_SEARCH_DIRS = [RAW_DIR, RAW_DIR / "archive"]


def _read_csv_if_exists(spark, name: str):
    for directory in RAW_SEARCH_DIRS:
        path = directory / name
        if path.exists():
            return spark.read.csv(str(path), header=True, inferSchema=True)
    return None


def run() -> None:
    """Load Instacart CSV files from data/raw and persist normalized parquet tables."""
    ensure_dirs()

    required_files = [
        "orders.csv",
        "order_products__prior.csv",
        "order_products__train.csv",
        "products.csv",
        "aisles.csv",
        "departments.csv",
    ]

    if use_pandas_engine():
        _run_pandas(required_files)
        return

    spark = get_spark("retail-ai-ingestion")
    loaded = {}
    for filename in required_files:
        frame = _read_csv_if_exists(spark, filename)
        if frame is None:
            print(f"Skipping missing raw file: {filename}")
            continue
        table_name = Path(filename).stem.replace("__", "_")
        output_path = PROCESSED_DIR / table_name
        frame.write.mode("overwrite").parquet(str(output_path))
        loaded[table_name] = output_path
        print(f"Ingested {filename} -> {output_path}")

    if not loaded:
        raise FileNotFoundError(
            "No Instacart CSV files found in data/raw or data/raw/archive. "
            "Add the Instacart CSVs and rerun."
        )


def _run_pandas(required_files: list[str]) -> None:
    loaded = {}
    for filename in required_files:
        path = next((directory / filename for directory in RAW_SEARCH_DIRS if (directory / filename).exists()), None)
        if path is None:
            print(f"Skipping missing raw file: {filename}")
            continue

        table_name = Path(filename).stem.replace("__", "_")
        output_path = PROCESSED_DIR / f"{table_name}.parquet"
        pd.read_csv(path).to_parquet(output_path, index=False)
        loaded[table_name] = output_path
        print(f"Ingested {filename} -> {output_path}")

    if not loaded:
        raise FileNotFoundError(
            "No Instacart CSV files found in data/raw or data/raw/archive. "
            "Add the Instacart CSVs and rerun."
        )
