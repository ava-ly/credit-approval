import pendulum
from airflow.decorators import dag, task

from src.data.make_dataset import run_dataset_creation
from src.features.build_features import run_feature_engineering
from src.models.train_model import train_and_save_model

# Define the requirements once and reuse them
VENV_REQUIREMENTS = ['pandas', 'scikit-learn', 'pyspark', 'xgboost', 'imbalanced-learn', 'pyarrow', 'joblib']

# Define our file paths centrally in the DAG
RAW_APP_PATH = "data/raw/application_record.csv"
RAW_CREDIT_PATH = "data/raw/credit_record.csv"
PRIMARY_DATASET_PATH = "data/processed/primary_dataset"
FEATURED_DATASET_PATH = "data/processed/featured_dataset"
FINAL_MODEL_PATH = "models/credit_risk_pipeline_final.joblib"

@dag(
    dag_id="credit_approval_taskflow_pipeline",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    tags=["production", "mlops", "taskflow"],
    doc_md="""
    ### End-to-End Credit Risk Model Training Pipeline (TaskFlow API)
    A modern DAG using decorators to automate the ML model creation process.
    """
)

def credit_approval_pipeline():
    """Defines the DAG's structure and tasks."""

    @task.virtualenv(
        task_id="make_dataset",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False
    )
    def make_dataset_task():
        """Runs the initial data processing."""
        from src.data.make_dataset import run_dataset_creation_with_paths
        output_path = run_dataset_creation_with_paths(
            app_path=RAW_APP_PATH, 
            credit_path=RAW_CREDIT_PATH, 
            output_path=PRIMARY_DATASET_PATH
        )
        return output_path
    
    @task.virtualenv(
        task_id="build_features",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False
    )
    def build_features_task(processed_data_path: str):
        """Runs feature engineering."""
        from src.features.build_features import run_feature_engineering_with_paths
        
        output_path = run_feature_engineering_with_paths(
            input_path=processed_data_path,
            output_path=FEATURED_DATASET_PATH
        )
        return output_path
    
    @task.virtualenv(
        task_id="train_model",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False
    )
    def train_model_task(featured_data_path: str):
        """Trains the final model."""
        from src.models.train_model import train_and_save_model_with_paths
        
        train_and_save_model_with_paths(
            input_path=featured_data_path,
            output_path=FINAL_MODEL_PATH
        )
        print(f"Model training and saving complete.")

    # Define the workflow by calling the functions
    processed_path = make_dataset_task()
    featured_path = build_features_task(processed_data_path=processed_path)
    train_model_task(featured_data_path=featured_path)

# Instantiates the DAG
credit_approval_pipeline()
dag = credit_approval_pipeline()