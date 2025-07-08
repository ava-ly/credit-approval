import os
from datetime import datetime
from airflow.decorators import dag, task

# Use container-mounted paths (matches your compose file)
BASE_PATH = "/opt/airflow"
RAW_APP_PATH = os.path.join(BASE_PATH, "data/raw/application_record.csv")
RAW_CREDIT_PATH = os.path.join(BASE_PATH, "data/raw/credit_record.csv")
PRIMARY_DATASET_PATH = os.path.join(BASE_PATH, "data/processed/primary_dataset")
FEATURED_DATASET_PATH = os.path.join(BASE_PATH, "data/processed/featured_dataset")
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "models/credit_risk_pipeline_final.joblib")

VENV_REQUIREMENTS = ['pandas', 'scikit-learn', 'pyspark', 'xgboost', 'imbalanced-learn', 'pyarrow', 'joblib']

@dag(
    dag_id="credit_approval_taskflow_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["production", "mlops"],
    doc_md="""Credit Risk Model Training Pipeline"""
)
def credit_approval_pipeline():
    
    @task(task_id="validate_paths")
    def validate_inputs():
        """Check if input files exist"""
        if not all(os.path.exists(p) for p in [RAW_APP_PATH, RAW_CREDIT_PATH]):
            raise FileNotFoundError("Missing input files!")
        return True

    @task.virtualenv(
        task_id="make_dataset",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False,
        python_version="3.8"
    )
    def make_dataset_task():
        from src.data.make_dataset import run_dataset_creation
        return run_dataset_creation(RAW_APP_PATH, RAW_CREDIT_PATH, PRIMARY_DATASET_PATH)

    @task.virtualenv(
        task_id="build_features",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False
    )
    def build_features_task(processed_data_path: str):
        from src.features.build_features import run_feature_engineering
        return run_feature_engineering(processed_data_path, FEATURED_DATASET_PATH)

    @task.virtualenv(
        task_id="train_model",
        requirements=VENV_REQUIREMENTS,
        system_site_packages=False
    )
    def train_model_task(featured_data_path: str):
        from src.models.train_model import train_and_save_model
        train_and_save_model(featured_data_path, FINAL_MODEL_PATH)

    # Task flow
    validation = validate_inputs()
    processed_path = make_dataset_task()
    featured_path = build_features_task(processed_data_path=processed_path)
    train_model_task(featured_data_path=featured_path)

dag = credit_approval_pipeline()