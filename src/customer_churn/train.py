
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path=None):
    """Load project configuration from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / "conf" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model(config_path=None):
    """Train, evaluate and log a churn model to MLflow."""
    config = load_config()
    if config_path is not None:
        config = load_config(config_path)

    data_config = config["data"]
    training = config["training"]
    mlflow_config = config["mlflow"]

    data_path = Path(data_config["path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    numeric_features = data_config["numeric_features"]
    categorical_features = data_config["categorical_features"]
    target_column = data_config["target_column"]
    feature_columns = numeric_features + categorical_features

    required_columns = set(feature_columns + [target_column])
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    data = data[feature_columns + [target_column]].copy()

    # Convert the target labels to binary values.
    data[target_column] = data[target_column].map(
        {"Yes": 1, "No": 0}
    )

    # Convert numeric fields safely; imputation happens in the pipeline.
    for column in numeric_features:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # Rows with an unknown target cannot be used for supervised training.
    data = data.dropna(subset=[target_column])

    X = data[feature_columns]
    y = data[target_column].astype(int)

    if data.empty or y.nunique() < 2:
        raise ValueError(
            "Training requires valid data from both churn classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training["test_size"],
        random_state=training["random_state"],
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=training["max_iter"]
            ),
        ),
    ])

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model_type": training["model_type"],
            "numeric_features": ",".join(numeric_features),
            "categorical_features": ",".join(
                categorical_features
            ),
            "random_state": training["random_state"],
            "test_size": training["test_size"],
            "max_iter": training["max_iter"],
        })

        mlflow.log_metrics(metrics)

        # Use floating-point numeric examples to reduce schema warnings.
        input_example = X_train.head(3).copy()
        input_example[numeric_features] = input_example[
            numeric_features
        ].astype("float64")

        # Trust only the specific NumPy type reported by the serializer.
        mlflow.sklearn.log_model(
            sk_model=model,
            name=mlflow_config["model_name"],
            input_example=input_example,
            skops_trusted_types=["numpy.dtype"],
        )

        print(f"Run ID: {run.info.run_id}")
        print(
            f"Model URI: runs:/{run.info.run_id}/"
            f"{mlflow_config['model_name']}"
        )
        print(f"Metrics: {metrics}")

    return model, metrics


if __name__ == "__main__":
    train_model()
