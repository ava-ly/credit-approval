import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Import the function to be tested
from src.data.make_dataset import generate_target_variable

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
    actual_df = generate_target_variable(input_df)

    # Assert that the actual output matches the expected output
    # Collecting data to compare them as lists of rows is easier than comparing DataFrames directly
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())
    assert actual_df.schema == expected_df.schema