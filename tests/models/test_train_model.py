import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline

# Import the function to be tested
from src.models.train_model import train_and_save_model

# Create a small, representative dummy dataframe to be returned by our mock
# It must contain all columns that the training script uses to determine feature types.
DUMMY_DATA = {
    'Risk_Flag': [0, 1, 0, 1],
    'ID': [1, 2, 3, 4],
    'DAYS_BIRTH': [-10000, -15000, -12000, -20000],
    'DAYS_EMPLOYED': [-500, -1000, 365243, -2000],
    'AGE_YEARS': [27.4, 41.1, 32.9, 54.8],
    'YEARS_EMPLOYED': [1.4, 2.7, 0.0, 5.5],
    'IS_UNEMPLOYED': [0, 0, 1, 0],
    'AMT_INCOME_TOTAL': [100000.0, 150000.0, 80000.0, 200000.0],
    'CNT_FAM_MEMBERS': [2.0, 1.0, 2.0, 4.0],
    'INCOME_PER_PERSON': [50000.0, 150000.0, 40000.0, 50000.0],
    'NAME_INCOME_TYPE': ['Working', 'State servant', 'Pensioner', 'Working'],
    'CODE_GENDER': ['M', 'F', 'F', 'M'],
    'OCCUPATION_TYPE': ['Managers', 'Core staff', None, 'Drivers']
}
DUMMY_DF = pd.DataFrame(DUMMY_DATA)

@patch('src.models.train_model.joblib.dump')
@patch('src.models.train_model.pd.read_parquet')
def test_train_and_save_model_orchestration(mock_read_parquet, mock_joblib_dump):
    """
    Tests the training script by mocking the file I/O operations (read and save).
    """
    # --- Arrange ---
    # Configure our mock for pd.read_parquet to return our dummy DataFrame
    mock_read_parquet.return_value = DUMMY_DF
    
    # --- Act ---
    train_and_save_model(
        input_path="fake/input/s3/path",
        output_path="fake/output/model.joblib",
        storage_options={"fake": "options"}
    )
    
    # --- Assert ---
    mock_read_parquet.assert_called_once_with("fake/input/s3/path", storage_options={"fake": "options"})
    mock_joblib_dump.assert_called_once()
    saved_pipeline = mock_joblib_dump.call_args[0][0]
    saved_path = mock_joblib_dump.call_args[0][1]
    assert isinstance(saved_pipeline, Pipeline)
    assert saved_path == "fake/output/model.joblib"