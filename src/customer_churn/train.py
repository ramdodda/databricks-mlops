from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def train_model(
    data_path: str,
    experiment_name: str = "/Shared/customer-churn-experiment",
    random_state: int = 42,
):
    """Train a churn classifier and log the model and metrics to MLflow."""

    data_file = Path(data_path)
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    data = pd.read_csv(data_file)

    required_columns = {"salary", "churn"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = data[["salary", "churn"]].copy()
    data["salary"] = pd.to_numeric(data["salary"], errors="coerce")
    data["churn"] = data["churn"].map({"Yes": 1, "No": 0})
    data = data.dropna()

    if data.empty or data["churn"].nunique() < 2:
        raise ValueError("Training requires valid data from both churn classes.")

    X = data[["salary"]]
    y = data["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("features", "salary")
        mlflow.log_param("random_state", random_state)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="customer_churn_model",
            input_example=X_train.head(3),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

    return model, metrics


if __name__ == "__main__":
    train_model("data/customer_churn.csv")
