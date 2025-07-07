from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Define file paths
FEATURED_DATA_PATH = Path("data/processed/featured_dataset.parquet")
FINAL_MODEL_PATH = "models/credit_risk_pipeline_final.joblib"

def train_and_save_model():
    """
    This function loads the latest featured data, 
    trains the pipeline and saves the result.
    """
    # Load the data
    df = pd.read_parquet(FEATURED_DATA_PATH)
    print('Data loaded successfully.')

    # Define the target and features
    y = df['Risk_Flag']
    X = df.drop(columns=['Risk_Flag', 'ID', 'DAYS_BIRTH', 'DAYS_EMPLOYED'])

    # Identify feature types for the preprocessor
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    print(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features.")

    # Define the final pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    final_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Train the pipeline on the entire dataset
    final_pipeline.fit(X, y)
    print('Model training complete.')

    # Save the trained pipeline artifact
    joblib.dump(final_pipeline, FINAL_MODEL_PATH)
    print(f"Model successfully saved to {FINAL_MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()