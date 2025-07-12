# Credit Approval with Data Engineering & MLOps
<!-- Badges Section -->
<p>
    <img src="https://github.com/ava-ly/credit-approval/actions/workflows/main.yml/badge.svg" alt="Build Status">
    <img src="https://img.shields.io/badge/MLOps_Docker-Airflow_2.8.1_|_PySpark_3.5-0073C6?logo=docker&logoColor=white" alt="Container">
    <a href="https://codecov.io/gh/ava-ly/credit-approval" >
        <img src="https://codecov.io/gh/ava-ly/credit-approval/graph/badge.svg?token=70U95XEZJ7" alt="Code Coverage">
    </a>
</p>

Designed and built an end-to-end MLOps pipeline with automated testing and deployment, orchestrated by Airflow, capable of processing data from cloud object storage (AWS S3) in a production-like environment.

## Project Structure

This structure makes the project easy to navigate, test, and maintain.

```
credit-approval/
├── .github/                      # For CI/CD workflows
│   └── workflows/
│       └── main.yml              # GitHub Actions workflow file
├── .gitignore                    # To ignore temp files, data, etc.
├── data/
│   ├── raw/                      # The original, immutable data
│   │   ├── application_record.csv
│   │   └── credit_record.csv
│   └── processed/                # Data after cleaning and merging
│       └── cleaned_application.parquet
├── dags/                         # Airflow DAG definitions
│   └── credit_approval_dag.py
├── notebooks/                    # Jupyter notebooks for EDA
│   └── 01-initial-exploration.ipynb
├── src/                          # Source code for the project
│   ├── data/
│   │   └── make_dataset.py       # Scripts to download, merge, clean data
│   ├── features/
│   │   └── build_features.py     # Scripts to create new features
│   ├── models/
│   │   ├── train_model.py        # Script to train the model
│   │   └── predict_model.py      # Script to make predictions (optional)
│   └── visualization/
│       └── visualize.py          # Scripts to create evaluation plots
├── tests/                        # Automated tests for the source code
│   ├── data/
│   │   └── test_make_dataset.py
│   ├── features/
│   │   └── test_build_features.py
│   └── models/
│       └── test_train_model.py
├── Dockerfile                    # To containerize the application/pipeline
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```


## Phase 1: Environment & Pipeline Setup ![icon-url]

**Goal:** Establish the project foundation and CI pipeline.

- Project Scaffolding: Create the folder structure outlined above.
- Version Control: Initialized a Git repository and configured a .gitignore file to exclude data, environment files, and other non-source assets.
- Dependency Management: Set up a Python virtual environment (venv) and defined all project dependencies in requirements.txt.
- CI/CD Pipeline: Implemented a Continuous Integration pipeline using GitHub Actions to automatically run tests and calculate code coverage on every push to the main branch.
- Orchestration Setup: Deployed a local Apache Airflow instance using Docker and Docker Compose for development and testing of automated workflows.

## Phase 2: Data Processing ![icon-url]

**Goal:** Process the raw data and create a clean, unified primary dataset.

1. Target Variable Engineering (TDD):
    - Write Test (`tests/data/test_make_dataset.py`): Create a small, sample PySpark DataFrame that mimics `credit_record.csv`. Write a test function `test_target_variable_creation` that asserts:
        - A client with a `STATUS` of '5' gets a `Risk_Flag` of 1.
        - A client with only 'C' and 'X' status gets a `Risk_Flag` of 0.
        - The output DataFrame has the correct columns (`ID`, `Risk_Flag`).
    - Run `pytest`: Watch it fail (RED).
    - Write Code (`src/data/make_dataset.py`): Implement the `generate_target_variable` function using PySpark to make the test pass (GREEN).
    - Refactor: Clean up the code.

2. Data Merging and Cleaning (TDD):
    - Write Test: In the same test file, add a test for the main data processing function. It should check that the application and credit data are merged correctly and that known data anomalies (e.g., `DAYS_EMPLOYED`) are handled.
    - Write Code: Implement the logic in `src/data/make_dataset.py` to pass the test. This script will be the **1st major task** in our Airflow pipeline.


## Phase 3: Exploratory Data Analysis (EDA) ![icon-url]

**Goal:** Understand the processed data, identify patterns, discover relationships.

- Create a new notebook (`notebooks/01_credit_risk_analysis.ipynb`)
- Load the Data: Use PySpark to read `../data/processed/primary_dataset`.
- Analyze Target Variable: Plot the distribution of `Risk_Flag`.
- Univariate Analysis: Explore single features.
- Bivariate Analysis: Explore the relationship between each feature and the `Risk_Flag`.


## Phase 4: Feature Engineering & Model Prototyping ![icon-url]

**Goal:** Create new, predictive features and experiment with various machine learning models to find the best performer.

1. Feature Engineering (TDD):
    - Write Test (`tests/features/test_build_features.py`): Create a test to check the feature creation logic.
    - Write Code (`src/features/build_features.py`): Implement the `create_new_features` function using PySpark to pass the test. This script will be the **2nd task** in our pipeline.

2. Model Training and Prototyping (Notebook):
    - Create a new notebook (`notebooks/02_model_prototyping.ipynb`) to find the best model.
    - Train multiple models: Logistic Regression, Random Forest, XGBoost.
    - Evaluate them: Using metrics like AUC, Precision, and Recall.
    - Apply class weighting, resampling (SMOTE) to handle class imbalance.
    - Select the best model: find its best hyperparameters.

## Phase 5: Productionalizing the MLOps Pipeline ![icon-url]

**Goal:** Translate the findings from the notebooks into an automated, end-to-end training pipeline orchestrated by Airflow.

- Create Production Training Script: Developed a clean, non-interactive Python script (`src/models/train_model.py`) that encapsulates the entire process of loading the featured data, defining the preprocessing and SMOTE pipeline, and training the final Random Forest model on all available data.
- Save Model Artifact: The script saves the final, trained pipeline object (including the preprocessor, SMOTE step, and model) as a single .joblib file in the `models/` directory.
- Build the **Airflow DAG**: Created a final Airflow DAG (`dags/credit_approval_full_pipeline.py`) that orchestrates the three main scripts: `make_dataset.py`, `build_features.py`, `train_model.py`.

## Phase 6: Cloud Integration with LocalStack (AWS S3) ![icon-url]

**Goal:** Refactor the pipeline to read and write to a cloud object storage system (emulated AWS S3) instead of the local filesystem. 

- Integrated LocalStack into the `docker-compose.yaml` setup to provide a local, fully-functional AWS S3 service running in a Docker container.
- Wrote a simple script to automatically create the necessary S3 "bucket" on LocalStack startup, demonstrating infrastructure provisioning.
- Modified the PySpark scripts (`make_dataset.py`, `build_features.py`) to connect to the S3 endpoint. The scripts now read their inputs from and write their outputs to S3 URIs.
- Configured the SparkSession with the necessary Hadoop S3A connector properties and JAR packages to enable communication with S3-compatible storage.
- Ran the entire Airflow DAG to verify that the pipeline executes successfully in a cloud-like architecture, moving data between stages entirely within the emulated S3 storage.

[icon-url]: https://github.com/ava-ly/credit-approval/blob/main/icon/ok-24.png?raw=true