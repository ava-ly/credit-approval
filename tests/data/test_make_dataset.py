import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Import the function to be tested
from src.data.make_dataset import generate_target_variable, process_full_dataset

# A pytest fixture to create a Spark session for testing
# This is better than creating a Spark session in each test
@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-spark-testing") \
        .getOrCreate()

def test_target_variable_generation(spark):
    """
    Tests the generate_target_variable function to ensure it correctly
    identifies the high-risk clients based on their credit history.
    """
    # Define the schema for the test input data
    credit_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("MONTHS_BALANCE", IntegerType(), True),
        StructField("STATUS", StringType(), True),
    ])

    # Create sample input data
    # Client 101 is a good client (never past due)
    # Client 102 is a "high-risk" client 
    # Client 103 is also good client (no loan)
    input_data = [
        (101, -1, 'C'), (101, -2, 'C'),
        (102, -1, '0'), (102, -2, '1'), (102, -3, '2'),
        (103, -5, 'X'),
    ]
    input_df = spark.createDataFrame(data=input_data, schema=credit_schema)

    # Define the expected output after the function runs
    expected_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("Risk_Flag", IntegerType(), False),
    ])
    expected_data = [
        (101, 0),   # good client -> risk_flag = 0
        (102, 1),   # high-risk client -> risk_flag = 1
        (103, 0),   # good client -> risk_flag = 0
    ]
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    # Call the function
    actual_df = generate_target_variable(spark, input_df)

    # Assert that the actual output matches the expected output
    # Collecting data to compare them as lists of rows is easier than comparing DataFrames directly
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())
    assert actual_df.schema == expected_df.schema

def test_full_dataset_processing(spark):
    """
    Tests the end-to-end data processing pipeline, including loading,
    target generation, merging, and basic cleaning.
    """
    # 1. Create sample raw data DataFrames
    app_schema = StructType([StructField("ID", IntegerType()), StructField("CODE_GENDER", StringType())])
    app_data = [(101, 'F'), (102, 'M'), (104, 'F')] # client 104 has no credit history
    app_df = spark.createDataFrame(app_data, app_schema)

    credit_schema = StructType([StructField("ID", IntegerType()), StructField("STATUS", StringType())])
    credit_data = [(101, 'C'), (102, '3')] # client 101 is good, 102 is bad
    credit_df = spark.createDataFrame(credit_data, credit_schema)

    # 2. Call the new main processing function we are about to create
    # This function should handle the merging and joining internally
    processed_df = process_full_dataset(spark, app_df, credit_df)

    # 3. Define the expectations for the final, processed DataFrame
    # It should contain clients in BOTH datasets and correctly 'Risk_Flag'
    expected_data = [
        (101, 'F', 0),
        (102, 'M', 1),
    ]

    # 4. Assertions
    # Check the count of rows
    assert processed_df.count() == 2

    # Check the data content
    actual_data = [(row.ID, row.CODE_GENDER, row.Risk_Flag) for row in processed_df.collect()]
    assert sorted(actual_data) == sorted(expected_data)

    # Check that the final DataFrame has the expected columns
    assert "Risk_Flag" in processed_df.columns
    assert "CODE_GENDER" in processed_df.columns