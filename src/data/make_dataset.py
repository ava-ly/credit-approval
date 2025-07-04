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