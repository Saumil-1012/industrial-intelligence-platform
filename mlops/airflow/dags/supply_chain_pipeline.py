"""
Apache Airflow DAG - Supply Chain ML Pipeline
Runs weekly: ingest -validate - featurize - materialize - retrain (if drift)

Milestones:
  Week 1: DAG skeleton
  Week 2: GE validation + Feast materialization steps - CURRENT
  Week 7: Evidently drift check + conditional retraining
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "industrial-intelligence",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def task_validate_raw_data(**context):
    """
    Step 1: Run Great Expectations on raw M5 data.
    Fails fast if data quality checks don't pass.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    import pandas as pd
    from great_expectations.expectations.m5_sales_suite import M5SalesValidator

    data_path = Path("/opt/airflow/data/supply_chain/sales_train_validation.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}. Run download.py first.")

    # Load a sample for fast validation (full validation in feature step)
    df = pd.read_csv(data_path, nrows=5000)
    validator = M5SalesValidator(df, dataset_name="m5_sales_train")
    report = validator.run_all()

    context["task_instance"].xcom_push(key="validation_report", value=report)

    if report["overall_status"] == "FAIL":
        raise ValueError(
            f"Raw data validation failed: {report['failed']} checks failed. "
            f"Pipeline halted - fix data quality issues first."
        )

    print(f"Data validation: {report['passed']}/{report['total_expectations']} passed")
    return report


def task_build_and_validate_features(**context):
    """
    Step 2: Feature engineering + GE feature validation.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    from modules.supply_chain.features.feature_engineering import build_features
    from great_expectations.expectations.m5_sales_suite import validate_features

    df = build_features(sample=False)
    report = validate_features(df)

    print(f"Feature validation: {report['passed']}/{report['total_expectations']} passed")
    context["task_instance"].xcom_push(key="feature_shape", value=str(df.shape))
    return {"shape": str(df.shape), "report": report}


def task_materialize_features(**context):
    """
    Step 3: Write features to Feast offline store (Parquet).
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    from feature_store.materialize import materialize_supply_chain

    df = materialize_supply_chain(sample=False)
    print(f"Materialized {len(df):,} rows to Feast offline store")


def task_check_drift(**context):
    """
    Step 4 (Week 7): Check Evidently drift score.
    Returns branch: 'retrain' or 'skip_retrain'.
    """
    # Week 7 milestone — placeholder returns skip for now
    print("Drift check: Evidently AI integration — Week 7 milestone")
    return "skip_retrain"


def task_retrain_model(**context):
    """
    Step 5: Retrain LightGBM models when drift detected.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    from modules.supply_chain.models.train import train_pipeline

    models, feature_cols, metrics = train_pipeline(sample=False, n_trials=50)
    print(f"Retrain complete. Metrics: {metrics}")


def task_skip_retrain(**context):
    """No retrain needed — drift below threshold."""
    print("No drift detected. Skipping retrain.")


with DAG(
    dag_id="supply_chain_weekly_pipeline",
    default_args=default_args,
    description="Weekly supply chain ML pipeline: validate → featurize → materialize → retrain",
    schedule="0 2 * * 1",  # Every Monday at 2am
    catchup=False,
    tags=["supply_chain", "ml", "weekly"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    validate_raw = PythonOperator(
        task_id="validate_raw_data",
        python_callable=task_validate_raw_data,
    )

    build_features = PythonOperator(
        task_id="build_and_validate_features",
        python_callable=task_build_and_validate_features,
    )

    materialize = PythonOperator(
        task_id="materialize_features",
        python_callable=task_materialize_features,
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=task_retrain_model,
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=task_skip_retrain,
    )

    # Pipeline: start → validate → features → materialize → drift check → branch
    start >> validate_raw >> build_features >> materialize >> check_drift
    check_drift >> [retrain, skip_retrain]
    [retrain, skip_retrain] >> end


#  Airline pipeline DAG 

def task_validate_airline_data(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    import pandas as pd
    from great_expectations.expectations.airline_suite import AirlineSuiteValidator
    from pathlib import Path

    data_dir = Path("/opt/airflow/data/airline")
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No airline CSVs found. Run data/airline/download.py first.")

    df = pd.read_csv(csv_files[0], nrows=10000)
    validator = AirlineSuiteValidator(df, dataset_name=csv_files[0].stem)
    report    = validator.run_all()

    if report["overall_status"] == "FAIL":
        raise ValueError(f"Airline data validation failed: {report['failed']} checks.")
    print(f"Airline validation: {report['passed']}/{report['total_expectations']} passed")


def task_train_airline_model(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    from modules.airline.models.train import train_pipeline
    train_pipeline(sample_frac=0.1, n_trials=20)
    print("Airline model retrain complete")


with DAG(
    dag_id="airline_weekly_pipeline",
    default_args=default_args,
    description="Weekly airline ML pipeline: validate → train",
    schedule="0 3 * * 1",   # Every Monday at 3am (1h after supply chain)
    catchup=False,
    tags=["airline", "ml", "weekly"],
) as airline_dag:

    a_start          = EmptyOperator(task_id="start")
    a_end            = EmptyOperator(task_id="end")
    a_validate       = PythonOperator(task_id="validate_airline_data",
                                       python_callable=task_validate_airline_data)
    a_train          = PythonOperator(task_id="train_airline_model",
                                       python_callable=task_train_airline_model)

    a_start >> a_validate >> a_train >> a_end
