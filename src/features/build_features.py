from pyspark.sql import functions as F

from src.utils.helpers import get_spark, load_yaml, project_path


def run() -> None:
    """Build model features from the processed Instacart table."""
    spark = get_spark("retail-ai-features")
    config = load_yaml(project_path("config/features.yaml"))["features"]

    input_path = project_path("data/processed/instacart_joined")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {input_path}")

    df = spark.read.parquet(str(input_path))

    user_features = df.groupBy("user_id").agg(
        F.countDistinct("order_id").alias("user_order_count"),
        F.avg("days_since_prior_order").alias("user_avg_days_between_orders"),
    )

    product_features = df.groupBy("product_id").agg(
        F.count("*").alias("product_order_count"),
        F.avg("reordered").alias("product_reorder_rate"),
    )

    features = (
        df.join(user_features, on="user_id", how="left")
        .join(product_features, on="product_id", how="left")
        .fillna(0)
    )

    selected_columns = (
        ["user_id", "product_id"]
        + config["numeric_columns"]
        + config["categorical_columns"]
        + [config["label_column"]]
    )
    selected_columns = [column for column in selected_columns if column in features.columns]

    output_path = project_path("data/features/instacart_features")
    features.select(*selected_columns).write.mode("overwrite").parquet(str(output_path))
    print(f"Wrote feature table -> {output_path}")

