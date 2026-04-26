import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.helpers import get_spark, load_yaml, project_path


def _build_model(model_config: dict) -> RandomForestClassifier:
    model_type = model_config["model"]["type"]
    params = model_config["model"].get("params", {})
    if model_type != "random_forest":
        raise ValueError(f"Unsupported model type: {model_type}")
    return RandomForestClassifier(**params)


def run() -> None:
    training_config = load_yaml(project_path("config/training.yaml"))["training"]
    feature_config = load_yaml(project_path("config/features.yaml"))["features"]
    model_config = load_yaml(project_path("config/model.yaml"))

    spark = get_spark("retail-ai-training")
    feature_path = project_path("data/features/instacart_features")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature dataset: {feature_path}")

    df = spark.read.parquet(str(feature_path)).limit(training_config["max_rows"]).toPandas()
    label_column = feature_config["label_column"]
    if label_column not in df.columns:
        raise ValueError(f"Label column not found in features: {label_column}")

    y = df[label_column].astype(int)
    X = df.drop(columns=[label_column])

    numeric_columns = [column for column in feature_config["numeric_columns"] if column in X.columns]
    categorical_columns = [
        column for column in feature_config["categorical_columns"] if column in X.columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", _build_model(model_config)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=y if y.nunique() > 1 else None,
    )

    mlflow.set_experiment(training_config["experiment_name"])
    with mlflow.start_run() as run_context:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {"accuracy": accuracy_score(y_test, predictions)}

        if hasattr(pipeline.named_steps["classifier"], "predict_proba") and y_test.nunique() > 1:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

        mlflow.log_params(model_config["model"].get("params", {}))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=training_config["registered_model_name"],
        )

        artifact_dir = project_path("data/predictions")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prediction": predictions, "actual": y_test.to_numpy()}).to_csv(
            artifact_dir / "validation_predictions.csv", index=False
        )
        Path(artifact_dir / "latest_metrics.json").write_text(json.dumps(metrics, indent=2))

        print(f"Training complete. MLflow run_id={run_context.info.run_id}, metrics={metrics}")

