import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Import the functions to be tested
from src.features.build_features import create_features, run_feature_engineering

# ==============================================================================
# Unit Test for Core Logic
# ==============================================================================
def test_create_features(spark):
    """Tests that new features are created correctly."""
    input_schema = StructType([
        StructField("DAYS_BIRTH", IntegerType()),
        StructField("DAYS_EMPLOYED", IntegerType()),
        StructField("AMT_INCOME_TOTAL", DoubleType()),
        StructField("CNT_FAM_MEMBERS", DoubleType()),
    ])
    input_data = [(-10950, -730, 100000.0, 2.0), (-16425, 365243, 80000.0, 4.0)]
    input_df = spark.createDataFrame(data=input_data, schema=input_schema)

    featured_df = create_features(input_df)
    
    results = featured_df.select("AGE_YEARS", "YEARS_EMPLOYED", "IS_UNEMPLOYED", "INCOME_PER_PERSON").collect()
    
    # Assertions for the first row
    assert round(results[0]["AGE_YEARS"]) == 30
    assert round(results[0]["YEARS_EMPLOYED"]) == 2
    assert results[0]["IS_UNEMPLOYED"] == 0
    assert results[0]["INCOME_PER_PERSON"] == 50000.0
    
    # Assertions for the second row
    assert round(results[1]["AGE_YEARS"]) == 45
    assert round(results[1]["YEARS_EMPLOYED"]) == 0
    assert results[1]["IS_UNEMPLOYED"] == 1
    assert results[1]["INCOME_PER_PERSON"] == 20000.0

# ==============================================================================
# Integration Test for Orchestration
# ==============================================================================
def test_run_feature_engineering_orchestration():
    # --- Arrange ---
    mock_spark = MagicMock(spec=SparkSession)
    mock_primary_df = MagicMock()
    mock_featured_df = MagicMock()
    
    mock_spark.read.parquet.return_value = mock_primary_df
    
    with patch('src.features.build_features.create_features') as mock_create_features:
        mock_create_features.return_value = mock_featured_df
        mock_featured_df.write.mode.return_value.parquet.return_value = None
        
        # --- Act ---
        run_feature_engineering(
            spark=mock_spark, 
            input_path="fake/input/path", 
            output_path="fake/output/path"
        )
        
        # --- Assert ---
        mock_spark.read.parquet.assert_called_once_with("fake/input/path")
        mock_create_features.assert_called_once_with(mock_primary_df)
        mock_featured_df.write.mode.assert_called_once_with("overwrite")
        mock_featured_df.write.mode.return_value.parquet.assert_called_once_with("fake/output/path")