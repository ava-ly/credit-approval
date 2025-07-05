import os
from pyspark.sql import SparkSession, DataFrame, functions as F

def configure_spark() -> SparkSession:
    """Initialize Spark with optimized settings for WSL2/Linux"""
    return SparkSession.builder \
        .appName("CreditRiskAnalysis") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

def generate_target_variable(credit_df: DataFrame) -> DataFrame:
    """Identify high-risk clients (no Windows-specific workarounds needed)"""
    return (
        credit_df
        .withColumn("STATUS", F.coalesce(F.col("STATUS").cast("string"), F.lit("0")))
        .withColumn("is_high_risk", F.when(F.col("STATUS").isin(["2", "3", "4", "5"]), 1).otherwise(0))
        .groupBy("ID")
        .agg(F.max("is_high_risk").alias("Risk_Flag"))
    )

def process_data(app_df: DataFrame, credit_df: DataFrame) -> DataFrame:
    """Join application data with risk flags"""
    return app_df.join(generate_target_variable(credit_df), "ID", "inner")

if __name__ == "__main__":
    spark = configure_spark()
    
    try:
        # Load data (use Linux-style paths)
        app_df = spark.read.csv("data/raw/application_record.csv", header=True, inferSchema=True)
        credit_df = spark.read.csv("data/raw/credit_record.csv", header=True, inferSchema=True)

        # Process and save
        process_data(app_df, credit_df) \
            .repartition(50) \
            .write \
            .mode("overwrite") \
            .parquet("data/processed/primary_dataset")

        print("Success! Output saved to data/processed/primary_dataset")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        spark.stop()