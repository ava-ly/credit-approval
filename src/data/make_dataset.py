from pyspark.sql import SparkSession, DataFrame, Window, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType

# CORE LOGIC FUNCTIONS (tested by pytest)
def generate_target_variable(spark: SparkSession, credit_df: DataFrame) -> DataFrame:
    """
    Identify high-risk clients
    
    Args:
        spark: The active SparkSession
        credit_df: DataFrame containing credit records

    Returns:
        A Spark DataFrame with two columns: 'ID', 'Risk_Flag'
    """
    high_risk_statuses = ['2', '3', '4', '5']

    df_risk_record = credit_df.withColumn(
        'is_high_risk',
        F.when(F.col('STATUS').isin(high_risk_statuses), 1).otherwise(0)
    )

    window_spec = Window.partitionBy('ID')

    df_risk_flag = df_risk_record.withColumn(
        'Risk_Flag',
        F.max('is_high_risk').over(window_spec)
    )

    intermediate_df = df_risk_flag.select('ID', 'Risk_Flag').distinct()

    final_schema = StructType([
        StructField('ID', IntegerType(), True),
        StructField('Risk_Flag', IntegerType(), False)
    ])

    final_df = spark.createDataFrame(
        intermediate_df.rdd,
        schema=final_schema
    )

    return final_df

def process_full_dataset(spark: SparkSession, app_df: DataFrame, credit_df: DataFrame) -> DataFrame:
    """Join application data with risk flags"""
    target_df = generate_target_variable(spark, credit_df)
    processed_df = app_df.join(target_df, on="ID", how="inner")
    return processed_df

# MAIN FUNCTION (called by Airflow)
def run_dataset_creation(spark: SparkSession, app_path: str, credit_path: str, output_path: str) -> str:
    """
    Main function to run the full dataset creation process.
    It takes a SparkSession and paths as input.
    """
    print(f"---(Make Dataset): Loading raw data from {app_path} and {credit_path}...")
    application_df = spark.read.csv(app_path, header=True, inferSchema=True)
    credit_record_df = spark.read.csv(credit_path, header=True, inferSchema=True)
    
    print("---(Make Dataset): Processing full dataset...")
    final_df = process_full_dataset(spark, application_df, credit_record_df)
    
    print(f"---(Make Dataset): Saving processed data to {output_path}...")
    final_df.write.mode("overwrite").parquet(output_path)
    
    print("---(Make Dataset): Data processing complete.---")
    return output_path

# --- The executable block now handles configuration and calls the logic ---
if __name__ == '__main__':
    # S3A configuration is now centralized here in the entrypoint
    spark = (
        SparkSession.builder
        .appName("CreditApprovalDataProcessing")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localstack:4566")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
        .getOrCreate()
    )
    
    APP_DATA_PATH = "s3a://credit-approval-data/raw/application_record.csv"
    CREDIT_DATA_PATH = "s3a://credit-approval-data/raw/credit_record.csv"
    OUTPUT_PATH = "s3a://credit-approval-data/processed/primary_dataset"
    
    run_dataset_creation(spark, APP_DATA_PATH, CREDIT_DATA_PATH, OUTPUT_PATH)
    
    spark.stop()