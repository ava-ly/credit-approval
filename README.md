# Credit Approval with Data Engineering & MLOps
[![Python Project CI](https://github.com/ava-ly/credit-approval/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/ava-ly/credit-approval/actions/workflows/main.yml) [![codecov](https://codecov.io/github/ava-ly/credit-approval/graph/badge.svg?token=70U95XEZJ7)](https://codecov.io/github/ava-ly/credit-approval)

## Project Structure

This structure separates concerns, making the project easy to navigate, test, and maintain.

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

1. Project Scaffolding: Create the folder structure outlined above.

2. Environment Setup:
    - Initialize a Git repository: `git init`.
    - Create `requirements.txt` with `pandas`, `numpy`, `pyspark`, `scikit-learn`, `pytest`, `apache-airflow`.
    - Set up a Python virtual environment and install dependencies.

3. Setup Initial CI Pipeline (GitHub Actions):
    - In `.github/workflows/main.yml`, create a basic workflow that triggers on every `push`.
    - This workflow should:
        - Check out the code.
        - Set up Python.
        - Install dependencies from `requirements.txt`.
        - Run `pytest`.
    
4. Setup Local Airflow & PySpark:
    - Install Airflow locally using Docker.
    - Ensure you can run a basic PySpark session.

## Phase 2: Test-Driven Development (TDD) and Data Processing ![icon-url]

**Goal:** Implement the data cleaning and feature engineering logic using TDD.

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
    - Write Code: Implement the logic in `src/data/make_dataset.py` to pass the test. This script will be the first major task in our Airflow pipeline.

3. Exploratory Data Analysis (Notebook):
    - Load the Data: Use PySpark to read `../data/processed/primary_dataset`.
    - Analyze Target Variable: Plot the distribution of `Risk_Flag`.
    - Univariate Analysis: Explore single features.
    - Bivariate Analysis: Explore the relationship between each feature and the `Risk_Flag`.

4. Feature Engineering (TDD):
    - Write Test (`tests/features/test_build_features.py`): Create a test to check the feature creation logic. For example, `test_age_creation` should assert that a `-DAYS_BIRTH` of -7300 results in an `AGE` of 20.
    - Write Code (`src/features/build_features.py`): Implement the `create_new_features` function using PySpark to pass the test. This script will be the second task in our pipeline.

5. Model Training and Prototyping (Notebook):
    - Create a new notebook (`notebooks/02-model-development.ipynb`) to find the best model.
    - Train multiple models: Logistic Regression, Random Forest, XGBoost.
    - Evaluate them: Using metrics like AUC, Precision, and Recall.
    - Select the best model: And find its best hyperparameters.

## Phase 3: Building the Airflow DAG

**Goal:** Orchestrate the tested scripts into an automated pipeline.


## Phase 4: Model Training & Deployment

**Goal:** Integrate the model training into the pipeline and prepare for deployment.

[icon-url]: https://github.com/ava-ly/credit-approval/blob/main/icon/ok-24.png?raw=true