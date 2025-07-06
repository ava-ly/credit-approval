import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

from src.features.build_features import create_features

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing"""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-spark-feature-testing") \
        .getOrCreate()

def test_feature_creation(spark):
    """Tests that new features (age, employment status, ratio) are created correctly."""
    # Define schema and create sample input data
    input_schema = StructType([
        StructField("ID", IntegerType()),
        StructField("DAYS_BIRTH", IntegerType()),
        StructField("DAYS_EMPLOYED", IntegerType()),
        StructField("AMT_INCOME_TOTAL", DoubleType()),
        StructField("CNT_FAM_MEMBERS", DoubleType()),
    ])
    input_data = [
        # Person 1: 30 years old, employed for 2 years
        (101, -10950, -730, 100000.0, 2.0),
        # Person 2: 45 years old, unemployed
        (102, -16425, 365243, 80000.0, 4.0),
    ]
    input_df = spark.createDataFrame(data=input_data, schema=input_schema)

    # Call the feature engineering function
    featured_df = create_features(input_df)

    # Define the expectations
    expected_data = [
        # Person 1
        (101, 30.0, 2.0, 0, 50000.0),
        # Person 2
        (102, 45.0, 0.0, 1, 20000.0),
    ]

    # Assertions to check the new columns
    actual_data = [(
        row.ID,
        round(row.AGE, 1), # prevent potential float precision issues
        round(row.YEARS_EMPLOYED, 1),
        row.IS_UNEMPLOYED,
        row.INCOME_PER_PERSON,
    ) for row in featured_df.select("ID", "AGE",
                                    "YEARS_EMPLOYED", 
                                    "IS_UNEMPLOYED",
                                    "INCOME_PER_PERSON").collect()
    ]
    assert sorted(actual_data) == sorted(expected_data)
    assert "AGE" in featured_df.columns
    assert "IS_UNEMPLOYED" in featured_df.columns