# Databricks notebook source
print("Hello, Databricks!")
print("Learning Databricks for MLOps and AI Engineering")

# COMMAND ----------


data = [
    ("Ravi", 25),
    ("Krishna", 30),
    ("Anil", 28)
]

columns = ["name", "age"]

df = spark.createDataFrame(data, columns)

display(df)


# COMMAND ----------


# 1. Select specific columns
df.select("name").show()

# 2. Filter records where age is greater than 26
df.filter(df.age > 26).show()

# 3. Add a new column
from pyspark.sql.functions import col

df2 = df.withColumn("age_next_year", col("age") + 1)

display(df2)


# COMMAND ----------


data = [
    (101, "Ravi", 50000, "Yes"),
    (102, "Krishna", 65000, "No"),
    (103, "Anil", 45000, "Yes"),
    (104, "Priya", 80000, "No"),
    (105, "Suresh", 55000, "Yes")
]

columns = ["customer_id", "name", "salary", "churn"]

customers = spark.createDataFrame(data, columns)

display(customers)


# COMMAND ----------



# COMMAND ----------


# Display the dataset schema
customers.printSchema()

# Count total customers
print("Total customers:", customers.count())

# Count customers by churn status
display(
    customers.groupBy("churn").count()
)


# COMMAND ----------


from pyspark.sql.functions import avg

customers.select(
    avg("salary").alias("average_salary")
).show()


# COMMAND ----------


customers.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer_churn_demo")

print("Customer data saved successfully!")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM customer_churn_demo;

# COMMAND ----------

display(spark.table("customer_churn_demo"))

# COMMAND ----------


from pyspark.sql.functions import col, count, when

# Check for missing values in each column
customers.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in customers.columns
]).show()

# Check duplicate customer records
print("Total records:", customers.count())
print("Distinct customer IDs:", customers.select("customer_id").distinct().count())


# COMMAND ----------


from pyspark.sql.functions import when, col

ml_data = customers.withColumn(
    "churn_label",
    when(col("churn") == "Yes", 1).otherwise(0)
)

display(ml_data)


# COMMAND ----------


from pyspark.sql.functions import col

# Select customer names and salaries
customers.select("name", "salary").show()

# Filter customers with salary greater than 50000
customers.filter(col("salary") > 50000).show()

# Create a new column with a 10% salary increase
customers.withColumn(
    "new_salary",
    col("salary") * 1.10
).show()

# Sort customers by salary, highest first
customers.orderBy(col("salary").desc()).show()


# COMMAND ----------


from pyspark.sql.functions import avg, max, min, count

# Average salary
customers.select(avg("salary").alias("average_salary")).show()

# Highest salary
customers.select(max("salary").alias("highest_salary")).show()

# Lowest salary
customers.select(min("salary").alias("lowest_salary")).show()

# Total customers
print("Total customers:", customers.count())


# COMMAND ----------


customers.groupBy("churn").agg(
    count("*").alias("customer_count"),
    avg("salary").alias("average_salary")
).show()


# COMMAND ----------


customers = spark.table("customebr_churn_demo")

display(customers)


# COMMAND ----------


orders_data = [
    (101, 2000),
    (102, 3500),
    (103, 1500),
    (106, 4000)
]

orders_columns = ["customer_id", "order_amount"]

orders = spark.createDataFrame(
    orders_data,
    orders_columns
)

display(orders)


# COMMAND ----------


inner_result = customers.join(
    orders,
    on="customer_id",
    how="inner"
)

display(inner_result)


# COMMAND ----------


left_result = customers.join(
    orders,
    on="customer_id",
    how="left"
)

display(left_result)


# COMMAND ----------


from pyspark.sql import functions as F

# Read the source Delta table
raw_df = spark.table("customer_churn_demo")

# Define the columns required for the pipeline
required_columns = [
    "customer_id",
    "name",
    "salary",
    "churn"
]

# Validate the schema
missing_columns = [
    c for c in required_columns
    if c not in raw_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Schema validation passed")
print("Input records:", raw_df.count())

display(raw_df)


# COMMAND ----------


# Remove duplicate customer IDs
# Reject rows with missing required values
# Reject invalid churn labels and negative salaries

silver_df = (
    raw_df
    .dropDuplicates(["customer_id"])
    .filter(F.col("customer_id").isNotNull())
    .filter(F.col("name").isNotNull())
    .filter(F.col("salary").isNotNull())
    .filter(F.col("salary") >= 0)
    .filter(F.col("churn").isin("Yes", "No"))
)

print("Raw records:", raw_df.count())
print("Clean records:", silver_df.count())
print("Rejected records:", raw_df.count() - silver_df.count())

display(silver_df)
b

# COMMAND ----------


from pyspark.sql import functions as F

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
    .select(
        "customer_id",
        "name",
        "salary",
        "churn",
        "churn_label",
        "salary_category"
    )
)

display(gold_df)


# COMMAND ----------


gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer_churn_gold")

print("Gold table created successfully")
display(spark.table("customer_churn_gold"))
