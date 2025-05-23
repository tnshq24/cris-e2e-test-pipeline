# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration and Validation
# MAGIC 
# MAGIC This notebook performs initial data exploration and validation for the MLOps pipeline.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

import yaml

config_path = "/Workspace/Repos/your-repo/mlops-project/config/pipeline_config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load data
data_path = config['data']['source_path']
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

print(f"Data shape: {df.count()} rows, {len(df.columns)} columns")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for missing values
missing_counts = df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
]).collect()[0].asDict()

print("Missing values per column:")
for col_name, missing_count in missing_counts.items():
    if missing_count > 0:
        percentage = (missing_count / df.count()) * 100
        print(f"  {col_name}: {missing_count} ({percentage:.2f}%)")

# COMMAND ----------

# Check for duplicates
duplicate_count = df.count() - df.dropDuplicates().count()
print(f"Duplicate rows: {duplicate_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Summary

# COMMAND ----------

# Get numerical columns
numerical_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double', 'float']]

# Statistical summary
df.select(numerical_cols).summary().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

# Convert to Pandas for visualization (sample if dataset is large)
if df.count() > 10000:
    sample_df = df.sample(0.1, seed=42).toPandas()
else:
    sample_df = df.toPandas()

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols[:4]):
    if i < len(axes):
        sample_df[col].hist(bins=30, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# COMMAND ----------

# Correlation matrix
correlation_matrix = sample_df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Validation Results

# COMMAND ----------

# Define validation rules
validation_results = {
    'total_rows': df.count(),
    'total_columns': len(df.columns),
    'duplicate_rows': duplicate_count,
    'missing_data_percentage': sum(missing_counts.values()) / (df.count() * len(df.columns)) * 100,
    'numerical_columns': len(numerical_cols),
    'categorical_columns': len(df.columns) - len(numerical_cols)
}

# Data quality score
quality_score = 100
if validation_results['duplicate_rows'] > 0:
    quality_score -= 10
if validation_results['missing_data_percentage'] > 5:
    quality_score -= 20
if validation_results['missing_data_percentage'] > 15:
    quality_score -= 30

validation_results['quality_score'] = quality_score

print("Data Validation Results:")
for key, value in validation_results.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# Save validation results
dbutils.fs.put(
    "/mnt/datalake/validation_results.json",
    str(validation_results),
    overwrite=True
)

print("✅ Data exploration and validation completed!")