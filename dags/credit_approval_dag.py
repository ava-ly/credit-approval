from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator # type: ignore

# This is the main definition of your workflow
with DAG(
    dag_id="hello_world_dag",                                 # The name of the DAG in the UI
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),       # A start date in the past
    catchup=False,                                            # If true, it would run for every day since the start_date
    schedule=None,                                            # This DAG runs only when manually triggered
    tags=["project_setup_test"],                              # Helps organize DAGs in the UI
) as dag:
    
    # This is your first task. It runs a simple shell command.
    task1 = BashOperator(
        task_id="say_hello",                                  # The name of the task
        bash_command="echo 'Hello from Airflow! My project setup is working.'",
    )

    # This is your second task.
    task2 = BashOperator(
        task_id="show_current_directory",
        bash_command="echo 'I am running in this directory:' && pwd",
    )

    # This line defines the dependency.
    # It tells Airflow that task1 must complete successfully before task2 can start.
    task1 >> task2