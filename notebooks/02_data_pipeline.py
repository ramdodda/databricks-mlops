# Databricks notebook source
from pyspark.sql import functions as F

# 1. Read source data
raw_df = spark.table("customer_churn_demo")

# 2. Validate schema
required_columns = ["customer_id", "name", "salary", "churn"]
missing = [c for c in required_columns if c not in raw_df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 3. Clean data (Silver)
silver_df = (
    raw_df
    .dropDuplicates(["customer_id"])
    .filter(F.col("customer_id").isNotNull())
    .filter(F.col("name").isNotNull())
    .filter(F.col("salary").isNotNull())
    .filter(F.col("salary") >= 0)
    .filter(F.col("churn").isin("Yes", "No"))
)

silver_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer_churn_silver")

# 4. Create ML-ready features (Gold)
gold_df = (
    silver_df
    .withColumn(
        "churn_label",
        F.when(F.col("churn") == "Yes", 1).otherwise(0)
    )
    .withColumn(
        "salary_category",
        F.when(F.col("salary") < 50000, "Low")
         .when(F.col("salary") <= 70000, "Medium")
         .otherwise("High")
    )
)

gold_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer_churn_gold")

print("Pipeline completed successfully")
print("Raw records:", raw_df.count())
print("Silver records:", silver_df.count())
print("Gold records:", gold_df.count())