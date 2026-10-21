# Databricks notebook source
from pyspark.sql import functions as F

gold_df = spark.table("customer_churn_gold")

print("Total records:", gold_df.count())
gold_df.printSchema()
display(gold_df)

# COMMAND ----------

# Select model features and target label
ml_df = gold_df.select(
    "salary",
    "churn_label"
)

display(ml_df)

# COMMAND ----------

# Convert the small practice dataset to Pandas
pdf = ml_df.toPandas()

X = pdf[["salary"]]
y = pdf["churn_label"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Churn distribution:")
print(y.value_counts())

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Actual labels:   ", y.tolist())
print("Predicted labels:", predictions.tolist())

# COMMAND ----------

# Load the Gold table
gold_df = spark.table("customer_churn_gold")

# Select the ML columns
ml_df = gold_df.select("salary", "churn_label")

# Convert to Pandas
pdf = ml_df.toPandas()

# Define features (X) and target (y)
X = pdf[["salary"]]
y = pdf["churn_label"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Churn distribution:")
print(y.value_counts())

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Actual labels:   ", y.tolist())
print("Predicted labels:", predictions.tolist())

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create or select an experiment
mlflow.set_experiment("/Shared/customer-churn-experiment")

# Start tracking this model run
with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log model parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("feature", "salary")
    mlflow.log_metric("training_accuracy", accuracy)

    # Log the trained model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="customer_churn_model"
    )

    print("Training accuracy:", accuracy)
    print("MLflow run ID:", mlflow.active_run().info.run_id)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = pdf[["salary"]]
y = pdf["churn_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# COMMAND ----------

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Evaluate the trained model on unseen test records
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(
    y_test, y_pred, zero_division=0
)
test_recall = recall_score(
    y_test, y_pred, zero_division=0
)
test_f1 = f1_score(
    y_test, y_pred, zero_division=0
)

with mlflow.start_run(
    experiment_id=mlflow.get_experiment_by_name(
        "/Shared/customer-churn-experiment"
    ).experiment_id
):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_records", len(X_test))

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)

    print("Test metrics logged to MLflow")
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1-score:", test_f1)

# COMMAND ----------

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

# Infer the expected input and output schemas
signature = infer_signature(
    X_train,
    model.predict(X_train)
)

# Create a sample input
input_example = X_train.head(2)

with mlflow.start_run(
    experiment_id=mlflow.get_experiment_by_name(
        "/Shared/customer-churn-experiment"
    ).experiment_id
):
    mlflow.sklearn.log_model(
        sk_model=model,
        name="customer_churn_model",
        signature=signature,
        input_example=input_example
    )

    print("Model logged with signature successfully")
    print(signature)

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd

run_id = "PASTE_MODEL_SIGNATURE_RUN_ID_HERE"

model_uri = f"runs:/{run_id}/customer_churn_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

new_customer = pd.DataFrame({
    "salary": [60000]
})

prediction = loaded_model.predict(new_customer)

print("Customer salary:", 60000)
print("Predicted churn label:", int(prediction[0]))
print(
    "Prediction:",
    "Likely to churn" if prediction[0] == 1
    else "Not predicted to churn"
)

# COMMAND ----------

import mlflow

experiment = mlflow.get_experiment_by_name(
    "/Shared/customer-churn-experiment"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=10
)

display(
    runs[["run_id", "status", "start_time", "artifact_uri"]]
)

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd

run_id = "7fa59439029249da82925554354bbaef"

model_uri = f"runs:/{run_id}/customer_churn_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully!")

# COMMAND ----------

import pandas as pd

# New customer with a salary of 60,000
new_customer = pd.DataFrame({
    "salary": [60000]
})

# Predict using the saved MLflow model
prediction = loaded_model.predict(new_customer)

print("Customer salary:", 60000)
print("Predicted churn label:", int(prediction[0]))
print(
    "Prediction:",
    "Likely to churn" if prediction[0] == 1
    else "Not predicted to churn"
)