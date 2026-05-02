import pandas as pd

from src.utils.helpers import get_spark, project_path, use_pandas_engine


PROCESSED_DIR = project_path("data/processed")


def _read_table(spark, name: str):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed table: {path}")
    return spark.read.parquet(str(path))


def run() -> None:
    """Create a training-ready joined order/product table."""
    if use_pandas_engine():
        _run_pandas()
        return

    from pyspark.sql import functions as F

    spark = get_spark("retail-ai-processing")

    orders = _read_table(spark, "orders")
    prior = _read_table(spark, "order_products_prior")
    train = _read_table(spark, "order_products_train")
    products = _read_table(spark, "products")

    order_products = prior.unionByName(train, allowMissingColumns=True)

    joined = (
        order_products.join(orders, on="order_id", how="inner")
        .join(products, on="product_id", how="left")
        .withColumn(
            "days_since_prior_order",
            F.coalesce(F.col("days_since_prior_order"), F.lit(0.0)),
        )
        .dropDuplicates(["order_id", "product_id"])
    )

    output_path = PROCESSED_DIR / "instacart_joined"
    joined.write.mode("overwrite").parquet(str(output_path))
    print(f"Wrote processed dataset -> {output_path}")


def _read_pandas_table(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed table: {path}")
    return pd.read_parquet(path)


def _run_pandas() -> None:
    orders = _read_pandas_table("orders")
    prior = _read_pandas_table("order_products_prior")
    train = _read_pandas_table("order_products_train")
    products = _read_pandas_table("products")

    order_products = pd.concat([prior, train], ignore_index=True)
    joined = (
        order_products.merge(orders, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .drop_duplicates(subset=["order_id", "product_id"])
    )
    joined["days_since_prior_order"] = joined["days_since_prior_order"].fillna(0.0)

    output_path = PROCESSED_DIR / "instacart_joined.parquet"
    joined.to_parquet(output_path, index=False)
    print(f"Wrote processed dataset -> {output_path}")
