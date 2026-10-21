import mlflow
import pandas as pd


def predict_churn(model_uri: str, salary: float) -> dict:
    """Load an MLflow model and predict churn for a customer's salary."""

    if salary < 0:
        raise ValueError("Salary cannot be negative.")

    model = mlflow.pyfunc.load_model(model_uri)

    input_data = pd.DataFrame({"salary": [salary]})
    prediction = int(model.predict(input_data)[0])

    return {
        "salary": salary,
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
    }


if __name__ == "__main__":
    # Replace with the actual MLflow run ID after training.
    run_id = "YOUR_MLFLOW_RUN_ID"
    model_uri = f"runs:/{run_id}/customer_churn_model"

    result = predict_churn(model_uri, salary=60000)
    print(result)
