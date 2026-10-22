
import os

import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Local model path by default; Docker overrides this with MODEL_URI
MODEL_URI = os.getenv(
    "MODEL_URI",
    "file:C:/Users/Ramakrishna Dodda/databricks-mlops/"
    "mlruns/1/models/m-f6dd84f3b42a417d90606eaefa5b9463/artifacts",
)


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn.",
    version="1.0.0",
)

model = None


# Request data schema
class Customer(BaseModel):
    tenure: float = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract: str
    InternetService: str
    PaymentMethod: str


# Load the trained MLflow model on application startup
@app.on_event("startup")
def load_model():
    global model

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Model loaded successfully from: {MODEL_URI}")
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        raise RuntimeError(
            f"Could not load MLflow model from {MODEL_URI}"
        ) from exc


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
    }


# Customer churn prediction endpoint
@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        # Match the model's expected input column names and data types
        customer_df = pd.DataFrame({
            "tenure": pd.Series(
                [customer.tenure], dtype="float64"
            ),
            "MonthlyCharges": pd.Series(
                [customer.MonthlyCharges], dtype="float64"
            ),
            "TotalCharges": pd.Series(
                [customer.TotalCharges], dtype="float64"
            ),
            "Contract": [customer.Contract],
            "InternetService": [customer.InternetService],
            "PaymentMethod": [customer.PaymentMethod],
        })

        prediction = int(model.predict(customer_df)[0])

        return {
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "message": (
                "Customer is likely to churn"
                if prediction == 1
                else "Customer is unlikely to churn"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
