import pytest
from unittest.mock import MagicMock, patch
from src.features.build_features import create_features, run_feature_engineering

@pytest.fixture
def mock_spark_builder():
    """Mock SparkSession builder pattern"""
    builder = MagicMock()
    spark = MagicMock()
    
    builder.appName.return_value = builder
    builder.getOrCreate.return_value = spark
    
    mock_df = MagicMock()
    spark.read.parquet.return_value = mock_df
    mock_df.write.mode.return_value.parquet.return_value = None
    
    return builder

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing create_features()"""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    
    data = [
        (-10000, 100, 2, 50000),  # Corrected: DAYS_BIRTH should be negative
        (-20000, -100, 3, 60000)
    ]
    columns = ["DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL"]
    
    return spark.createDataFrame(data, columns)

def test_create_features(sample_dataframe):
    """Test the core feature creation logic"""
    featured_df = create_features(sample_dataframe)
    results = featured_df.collect()
    
    # Verify calculations
    assert abs(results[0]["AGE"] - (10000/365)) < 0.001  # 10000 days → ~27.397 years
    assert results[0]["IS_UNEMPLOYED"] == 1  # DAYS_EMPLOYED > 0
    assert abs(results[1]["YEARS_EMPLOYED"] - (100/365)) < 0.001  # 100 days employed
    assert results[0]["INCOME_PER_PERSON"] == 25000

def test_run_feature_engineering(mock_spark_builder):
    """
    Test the main orchestration function
    We mock the transformation function itself (`create_features`) for this test.
    """
    with patch('pyspark.sql.SparkSession.builder', new=mock_spark_builder), \
         patch('src.features.build_features.create_features') as mock_create_features:
        
        # --- Arrange ---
        mock_spark = mock_spark_builder.getOrCreate.return_value

        # DataFrame returned by spark.read.parquet
        mock_input_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_input_df
        
        # DataFrame returned by the mocked create_features
        mock_output_df = mock_input_df 
        mock_create_features.return_value = mock_output_df

        # Configure the write mock on the DataFrame that will be written
        mock_output_df.write = MagicMock()
        mock_output_df.write.mode.return_value.parquet.return_value = None
        
        # --- Act ---
        # Run the function we are testing
        output_path = run_feature_engineering()
        
        # --- Assert ---
        # 1. Was spark.read.parquet called correctly?
        mock_spark.read.parquet.assert_called_once_with("data/processed/primary_dataset")
        
        # 2. Was our create_features function called with the DataFrame we read?
        mock_create_features.assert_called_once_with(mock_input_df)
        
        # 3. Was the write chain called on the result of create_features?
        mock_output_df.write.mode.assert_called_once_with("overwrite")
        mock_output_df.write.mode.return_value.parquet.assert_called_once_with("data/processed/featured_dataset")