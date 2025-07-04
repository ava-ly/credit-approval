from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType

def generate_target_variable(spark: SparkSession, credit_df: DataFrame) -> DataFrame:
    """
    Engineers the target variable 'Risk_Flag' from the credit record data

    A client is flagged as high-risk (1) if they have ever had a payment
    that was 60 or more days overdue (status '2', '3', '4', '5').
    Otherwise, they are considered low-risk (0).

    Args:
        credit_df: The Spark DataFrame containing credit record data.

    Returns:
        A Spark DataFrame with two columns: 'ID' and 'Risk_Flag'.
    """
    # Define the statuses that indicate a high-risk payment
    high_risk_statuses = ['2', '3', '4', '5']

    # Create a temporary column 'is_high_risk_record'
    # It will be 1 if the STATUS is in our high-risk list, otherwise 0
    df_with_risk_record = credit_df.withColumn(
        "is_high_risk_record",
        F.when(F.col("STATUS").isin(high_risk_statuses), 1).otherwise(0)
    )

    # Use a Window function to look at all records for a given ID
    # Partition data by ID so the max() function works per-client.
    window_spec = Window.partitionBy("ID")

    # Create the final 'risk_flag'
    # If the maximum value of 'is_high_risk_record' for a client is 1,
    # it means they had at least 1 high-risk payment.
    df_with_risk_flag = df_with_risk_record.withColumn(
        'Risk_Flag',
        F.max("is_high_risk_record").over(window_spec)
    )

    intermediate_df = df_with_risk_flag.select("ID", "Risk_Flag").distinct()

    final_schema = StructType([
        StructField("ID", IntegerType(), True),         # ID can be nullable
        StructField("Risk_Flag", IntegerType(), False)  # Risk_Flag cannot be null
    ])

    final_df = spark.createDataFrame(
        intermediate_df.rdd,
        schema=final_schema
    )

    return final_df

def process_full_dataset(spark: SparkSession, app_df: DataFrame, credit_df: DataFrame) -> DataFrame:
    """
    Orchestrates the full data processing pipeline.

    1. Generates the target variable from credit data
    2. Joins the target variable with the application data.

    Args:
        spark: the active SparkSession.
        app_df: DataFrame with application records.
        credit_df: DataFrame with credit history records.

    Returns:
        A single, cleaned DataFrame ready for feature engineering.
    """
    # Step 1: Use the tested function to get the risk flag
    target_df = generate_target_variable(spark, credit_df)

    # Step 2: Join the application data with the risk flag
    # Use inner join to keep only clients present in both datasets
    processed_df = app_df.join(target_df, on="ID", how="inner")

    return processed_df

# This block allows the script to run from the command line
if __name__ == '__main__':
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("CreditApprovalDataProcessing") \
        .getOrCreate()
    
    # 2. Define the file paths
    APP_DATA_PATH = 'data/raw/application_record.csv'
    CREDIT_DATA_PATH = 'data/raw/credit_record.csv'
    OUTPUT_PATH = 'data/processed/primary_dataset'

    # 3. Load the raw data
    print('Loading raw data...')
    application_df = spark.read.csv(APP_DATA_PATH, header=True, inferSchema=True)
    credit_record_df = spark.read.csv(CREDIT_DATA_PATH, header=True, inferSchema=True)

    # 4. Process the data using our functions
    print('Processing full dataset...')
    final_df = process_full_dataset(spark, application_df, credit_record_df)

    # 5. Save the processed data as a Parquet file
    print(f"Saving processed data to {OUTPUT_PATH}...")
    final_df.write.mode("overwrite").parquet(OUTPUT_PATH)

    print('Data processing complete.')

    # 6. Stop the Spark Session
    spark.stop()