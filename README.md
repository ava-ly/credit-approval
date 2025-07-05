# Predicting Credit Card Approval with Data Engineering & MLOps Mindset

## Professional Project Structure

This structure separates concerns, making the project easy to navigate, test, and maintain.

```
credit-card-approval/
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
├── notebooks/                    # Jupyter notebooks for exploration (EDA)
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
├── tests/                        # Automated tests for your source code
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


## **Phase 1: Environment & Pipeline Setup**

**Goal:** Establish the project foundation and CI pipeline.

*   **Task 1.1: Project Scaffolding:** Create the folder structure outlined above.
*   **Task 1.2: Environment Setup:**
    *   Initialize a Git repository: `git init`.
    *   Create `requirements.txt` with `pandas`, `pyspark`, `scikit-learn`, `pytest`, `apache-airflow`.
    *   Set up a Python virtual environment and install dependencies.
*   **Task 1.3: Setup Initial CI Pipeline (GitHub Actions):**
    *   In `.github/workflows/main.yml`, create a basic workflow that triggers on every `push`.
    *   This workflow should:
        1.  Check out the code.
        2.  Set up Python.
        3.  Install dependencies from `requirements.txt`.
        4.  Run `pytest`. (It will find no tests and pass, which is fine for now).
*   **Task 1.4: Setup Local Airflow & PySpark:**
    *   Install Airflow locally (using Docker is highly recommended for a clean setup).
    *   Ensure you can run a basic PySpark session.

## **Phase 2: Test-Driven Data Processing**

**Goal:** Implement the data cleaning and feature engineering logic using TDD.

*   **Task 2.1: Target Variable Engineering (TDD):**
    1.  **Write Test (`tests/data/test_make_dataset.py`):** Create a small, sample PySpark DataFrame that mimics `credit_record.csv`. Write a test function `test_target_variable_creation` that asserts:
        *   A client with a `STATUS` of '5' gets a `Risk_Flag` of 1.
        *   A client with only 'C' and 'X' status gets a `Risk_Flag` of 0.
        *   The output DataFrame has the correct columns (`ID`, `Risk_Flag`).
    2.  **Run `pytest`:** Watch it fail (RED).
    3.  **Write Code (`src/data/make_dataset.py`):** Implement the `generate_target_variable` function using PySpark to make the test pass (GREEN).
    4.  **Refactor:** Clean up the code.
*   **Task 2.2: Data Merging and Cleaning (TDD):**
    1.  **Write Test:** In the same test file, add a test for the main data processing function. It should check that the application and credit data are merged correctly and that known data anomalies (e.g., `DAYS_EMPLOYED`) are handled.
    2.  **Write Code:** Implement the logic in `src/data/make_dataset.py` to pass the test. This script will be the first major task in our Airflow pipeline.
*   **Task 2.3: Feature Engineering (TDD):**
    1.  **Write Test (`tests/features/test_build_features.py`):** Create a test to check the feature creation logic. For example, `test_age_creation` should assert that a `-DAYS_BIRTH` of -7300 results in an `AGE` of 20.
    2.  **Write Code (`src/features/build_features.py`):** Implement the `create_new_features` function using PySpark to pass the test. This script will be the second task in our pipeline.

## **Phase 3: Building the Airflow DAG**

**Goal:** Orchestrate the tested scripts into an automated pipeline.

*   **Task 3.1: Create the DAG file (`dags/credit_approval_dag.py`):**
    *   Define a new DAG object with a schedule (e.g., run daily).
    *   Define Airflow tasks using `BashOperator` or `PythonOperator`.
*   **Task 3.2: Define DAG Tasks:**
    1.  **Task 1 (`make_dataset_task`):** A task that executes the script `src/data/make_dataset.py`. It will take the raw CSVs as input and output a processed Parquet file to `data/processed/`.
    2.  **Task 2 (`build_features_task`):** A task that executes `src/features/build_features.py`. It depends on Task 1 finishing successfully.
    3.  **Task 3 (`train_model_task`):** A task that runs `src/models/train_model.py`. It depends on Task 2.
    4.  **Task 4 (`evaluate_model_task`):** A task that runs a script to generate evaluation charts/reports. It depends on Task 3.
*   **Task 3.3: Set Dependencies:** In the DAG file, define the execution order:
    `make_dataset_task >> build_features_task >> train_model_task >> evaluate_model_task`

## **Phase 4: Model Training & Deployment **

**Goal:** Integrate the model training into the pipeline and prepare for deployment.

*   **Task 4.1: Script the Model Training:**
    *   In `src/models/train_model.py`, write a script that:
        *   Loads the feature-engineered data from `data/processed/`.
        *   Performs final preprocessing (encoding, scaling).
        *   Handles class imbalance (e.g., using SMOTE on the training set).
        *   Trains an XGBoost model.
        *   Saves the trained model object (e.g., as a `.pkl` file) and evaluation metrics (e.g., as a `.json` file) to a `models/` directory.
*   **Task 4.2: Containerize with Docker (CD part):**
    *   Write a `Dockerfile`. This file is a recipe for creating a standard, isolated environment containing the code, dependencies, and Airflow configuration.
    *   This makes the pipeline portable and ensures it runs the same way on the machine as it does in the cloud.
*   **Task 4.3: "Deploying" the DAG:**
    *   For a production setup, the CD pipeline (e.g., GitHub Actions) would automatically copy the updated DAG file to the production Airflow server's `dags/` folder whenever you merge to the `main` branch.
