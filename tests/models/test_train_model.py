import pytest
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from src.models.train_model import train_and_save_model

# This fixture creates a temporary, isolated file system for the test to run in.
@pytest.fixture
def temp_project_dir(tmp_path):
    """Creates a temporary project structure with dummy data."""
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    
    # Create subdirectories
    data_processed_dir = tmp_path / "data" / "processed"
    data_processed_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Create dummy featured data
    dummy_data = {
        'ID': [1, 2, 3, 4], 'DAYS_BIRTH': [-10000, -15000, -12000, -20000],
        'DAYS_EMPLOYED': [-500, -1000, 365243, -2000], 'Risk_Flag': [0, 1, 0, 1],
        'AGE_YEARS': [27.4, 41.1, 32.9, 54.8], 'YEARS_EMPLOYED': [1.4, 2.7, 0.0, 5.5],
        'IS_UNEMPLOYED': [0, 0, 1, 0], 'INCOME_PER_PERSON': [50000.0, 75000.0, 25000.0, 100000.0],
        'NAME_INCOME_TYPE': ['Working', 'Commercial associate', 'Pensioner', 'Working'],
        'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary / secondary special', 'Higher education','Secondary / secondary special'],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', 'N'], 'FLAG_OWN_REALTY': ['Y', 'Y', 'N', 'N'],
        'CNT_CHILDREN': [0, 1, 0, 2], 'AMT_INCOME_TOTAL': [100000, 150000, 50000, 200000],
        'NAME_FAMILY_STATUS': ['Married', 'Single / not married', 'Married', 'Married'],
        'NAME_HOUSING_TYPE': ['House / apartment', 'With parents', 'House / apartment', 'Rented apartment'],
        'FLAG_MOBIL': [1, 1, 1, 1], 'FLAG_WORK_PHONE': [1, 0, 0, 1], 'FLAG_PHONE': [0, 0, 1, 1],
        'FLAG_EMAIL': [1, 0, 1, 0], 'OCCUPATION_TYPE': ['Managers', 'Sales staff', None, 'Drivers'],
        'CNT_FAM_MEMBERS': [2.0, 1.0, 2.0, 4.0], 'CODE_GENDER': ['F', 'M', 'F', 'M']
    }
    dummy_df = pd.DataFrame(dummy_data)
    parquet_path = data_processed_dir / "featured_dataset.parquet"
    dummy_df.to_parquet(parquet_path)
    
    return tmp_path

def test_train_and_save_model(temp_project_dir, monkeypatch):
    """
    An integration test for the model training script.
    It checks if the script runs and produces a valid model artifact.
    """
    # --- Arrange ---
    # Use monkeypatch to change the current working directory to our temp directory.
    # This makes all relative paths like 'data/processed' work inside the test.
    monkeypatch.chdir(temp_project_dir)

    test_data_path = temp_project_dir / "data/processed/featured_dataset.parquet"
    test_model_path = temp_project_dir / "models/credit_risk_pipeline_final.joblib"

    monkeypatch.setattr("src.models.train_model.FEATURED_DATA_PATH", test_data_path)
    monkeypatch.setattr("src.models.train_model.FINAL_MODEL_PATH", test_model_path)
    
    # --- Act ---
    # Run the function that orchestrates the training.
    train_and_save_model()
    
    # --- Assert ---
    assert test_model_path.exists()

    # Verify the model can be loaded
    loaded_model = joblib.load(test_model_path)
    assert isinstance(loaded_model, Pipeline)
    
    # Test prediction capability
    dummy_input_df = pd.read_parquet(test_data_path)
    X_dummy = dummy_input_df.drop(columns=['Risk_Flag', 'ID', 'DAYS_BIRTH', 'DAYS_EMPLOYED'])
    
    try:
        prediction = loaded_model.predict(X_dummy)
        assert len(prediction) == len(X_dummy)
    except Exception as e:
        pytest.fail(f"Model prediction failed with an exception: {e}")