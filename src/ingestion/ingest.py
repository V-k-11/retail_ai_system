from pathlib import Path

from src.utils.helpers import ensure_dirs, get_spark, project_path


RAW_DIR = project_path("data/raw")
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
    spark = get_spark("retail-ai-ingestion")

    required_files = [
        "orders.csv",
        "order_products__prior.csv",
        "order_products__train.csv",
        "products.csv",
        "aisles.csv",
        "departments.csv",
    ]

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
