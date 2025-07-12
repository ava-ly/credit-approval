import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def train_and_save_model(input_path: str, output_path: str, storage_options: dict = None):
    """
    This function loads the latest featured data, 
    trains the pipeline and saves the result.
    """
    print(f"Loading featured data from: {input_path}")
    df = pd.read_parquet(input_path, storage_options=storage_options)
    print("Data loaded successfully.")

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
    joblib.dump(final_pipeline, output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == '__main__':
    FEATURED_DATA_PATH = "s3a://credit-approval-data/processed/featured_dataset"
    FINAL_MODEL_PATH = "models/credit_risk_pipeline_final.joblib"
    S3_STORAGE_OPTIONS = {
        "key": "test",
        "secret": "test",
        "client_kwargs": {"endpoint_url": "http://localstack:4566"}
    }
    train_and_save_model(FEATURED_DATA_PATH, FINAL_MODEL_PATH, S3_STORAGE_OPTIONS)