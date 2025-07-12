# In tests/data/test_make_dataset.py
import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Import the functions to be tested
from src.data.make_dataset import generate_target_variable, run_dataset_creation

# ==============================================================================
# Unit Test for Core Logic
# ==============================================================================
def test_generate_target_variable(spark):
    """
    Tests the generate_target_variable function to ensure it correctly
    identifies high-risk clients based on their credit history.
    """
    credit_schema = StructType([
        StructField("ID", IntegerType()),
        StructField("MONTHS_BALANCE", IntegerType()),
        StructField("STATUS", StringType()),
    ])
    input_data = [
        (101, -1, 'C'), (101, -2, 'C'),
        (102, -1, '0'), (102, -2, '1'), (102, -3, '2'), 
        (103, -5, 'X'),
    ]
    input_df = spark.createDataFrame(data=input_data, schema=credit_schema)

    expected_schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("Risk_Flag", IntegerType(), False),
    ])
    expected_data = [(101, 0), (102, 1), (103, 0)]
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    actual_df = generate_target_variable(spark, input_df)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())
    assert actual_df.schema == expected_df.schema

# ==============================================================================
# Integration Test for Orchestration
# ==============================================================================
def test_run_dataset_creation_orchestration():
    """Tests the orchestration logic by passing in mock objects."""
    # --- Arrange ---
    mock_spark = MagicMock(spec=SparkSession) # Create a mock SparkSession
    mock_app_df = MagicMock()
    mock_credit_df = MagicMock()
    mock_final_df = MagicMock()
    
    mock_spark.read.csv.side_effect = [mock_app_df, mock_credit_df]
    mock_final_df.write.mode.return_value.parquet.return_value = None

    # We can even mock the inner function to isolate the test even more
    with patch('src.data.make_dataset.process_full_dataset') as mock_process_full_dataset:
        mock_process_full_dataset.return_value = mock_final_df
        
        # --- Act ---
        run_dataset_creation(
            spark=mock_spark,
            app_path="fake/app/path",
            credit_path="fake/credit/path",
            output_path="fake/output/path"
        )
        
        # --- Assert ---
        mock_spark.read.csv.assert_any_call("fake/app/path", header=True, inferSchema=True)
        mock_spark.read.csv.assert_any_call("fake/credit/path", header=True, inferSchema=True)
        mock_process_full_dataset.assert_called_once_with(mock_spark, mock_app_df, mock_credit_df)
        mock_final_df.write.mode.assert_called_once_with("overwrite")
        mock_final_df.write.mode.return_value.parquet.assert_called_once_with("fake/output/path")