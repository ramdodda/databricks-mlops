from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path=None):
    """Load project configuration from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / "conf" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model(config_path=None):
    """Train a churn classifier and log the run to MLflow."""
    config = load_config(config_path)

    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    feature_columns = config["data"]["feature_columns"]
    target_column = config["data"]["target_column"]

    required_columns = set(feature_columns + [target_column])
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = data[feature_columns + [target_column]].copy()
    data[target_column] = data[target_column].map({"Yes": 1, "No": 0})
    data = data.dropna()

    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()

    if data.empty or data[target_column].nunique() < 2:
        raise ValueError("Training requires valid data from both churn classes.")

    X = data[feature_columns]
    y = data[target_column].astype(int)

    training = config["training"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training["test_size"],
        random_state=training["random_state"],
        stratify=y,
    )

    model = LogisticRegression(max_iter=training["max_iter"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }

        mlflow.log_params({
            "model_type": training["model_type"],
            "features": ",".join(feature_columns),
            "random_state": training["random_state"],
            "test_size": training["test_size"],
        })
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name=config["mlflow"]["model_name"],
            input_example=X_train.head(3),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

    return model, metrics


if __name__ == "__main__":
    train_model()
