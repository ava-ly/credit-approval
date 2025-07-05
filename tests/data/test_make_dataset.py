import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from src.data.make_dataset import generate_target_variable, process_full_dataset

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-spark-testing") \
        .getOrCreate()

def test_target_variable_generation(spark):
    """Test risk flag generation logic"""
    credit_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("STATUS", StringType(), True),
    ])

    input_data = [
        (101, 'C'), (101, 'C'),  # Good client
        (102, '0'), (102, '2'),   # High-risk client
        (103, 'X'),                # No loan
    ]
    input_df = spark.createDataFrame(input_data, credit_schema)

    expected_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("Risk_Flag", IntegerType(), False),
    ])
    expected_data = [(101, 0), (102, 1), (103, 0)]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    actual_df = generate_target_variable(spark, input_df)
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_full_dataset_processing(spark):
    """Test end-to-end data processing"""
    app_schema = StructType([
        StructField("ID", IntegerType()),
        StructField("CODE_GENDER", StringType())
    ])
    app_data = [(101, 'F'), (102, 'M')]
    app_df = spark.createDataFrame(app_data, app_schema)

    credit_schema = StructType([
        StructField("ID", IntegerType()),
        StructField("STATUS", StringType())
    ])
    credit_data = [(101, 'C'), (102, '3')]
    credit_df = spark.createDataFrame(credit_data, credit_schema)

    processed_df = process_full_dataset(spark, app_df, credit_df)
    
    expected_data = [(101, 'F', 0), (102, 'M', 1)]
    actual_data = [(row.ID, row.CODE_GENDER, row.Risk_Flag) for row in processed_df.collect()]
    assert sorted(actual_data) == sorted(expected_data)