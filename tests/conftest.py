import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_spark_builder():
    """Mock SparkSession builder pattern"""
    builder = MagicMock()
    spark = MagicMock()
    
    # Mock the method chain: SparkSession.builder.appName().getOrCreate()
    builder.appName.return_value = builder
    builder.getOrCreate.return_value = spark
    
    # Mock DataFrame methods
    mock_df = MagicMock()
    spark.read.csv.return_value = mock_df
    mock_df.withColumn.return_value = mock_df
    mock_df.select.return_value = mock_df
    mock_df.distinct.return_value = mock_df
    
    return builder