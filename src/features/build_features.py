from pyspark.sql import DataFrame, SparkSession, functions as F

# CORE LOGIC FUNCTION (tested by pytest)
def create_features(df: DataFrame) -> DataFrame:
    """
    Creates new features based on EDA findings
    - AGE: age of applicants in years.
    - YEARS_EMPLOYED: Years of employment.
    - IS_UNEMPLOYED: binary flag for employment.
    - INCOME_PER_PERSON: Income per family member.
    """
    featured_df = (
        df.withColumn("AGE", -F.col("DAYS_BIRTH") / 365.0)
            .withColumn("IS_UNEMPLOYED", 
                        F.when(F.col("DAYS_EMPLOYED") > 0, 1)
                        .otherwise(0))
            .withColumn("YEARS_EMPLOYED",
                        F.when(F.col("DAYS_EMPLOYED") > 0, 0)
                        .otherwise(-F.col("DAYS_EMPLOYED") / 365.0))
            .withColumn("INCOME_PER_PERSON", 
                        F.col("AMT_INCOME_TOTAL") / F.col("CNT_FAM_MEMBERS"))
    )
    return featured_df

# MAIN FUNCTION (called by Airflow)
def run_feature_engineering(
    input_path: str = "s3a://credit-approval-data/processed/primary_dataset", 
    output_path: str = "s3a://credit-approval-data/processed/featured_dataset"
    ) -> str:
    """
    Main function to run the feature engineering process.

    Returns:
        The path to the output directory.
    """
    spark = (
        SparkSession.builder
        .appName("FeatureEngineering")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localstack:4566")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    primary_df = spark.read.parquet(input_path)

    featured_df = create_features(primary_df)

    featured_df.write.mode("overwrite").parquet(output_path)

    print("--- (Build Features): Feature engineering complete. ---")
    spark.stop()

    return output_path

if __name__ == '__main__':
    run_feature_engineering()
