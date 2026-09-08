"""
Airflow DAG - Weekly Drift Monitoring
Runs Evidently drift checks on both modules every Monday at 4am.
Auto-triggers module retraining DAGs when drift > threshold.

Architecture:
  drift_check_supply_chain --> [retrain_supply | skip_supply]
  drift_check_airline  -->[retrain_airline | skip_airline]
Both run in parallel - airline doesn't block supply chain.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner":            "industrial-intelligence",
    "depends_on_past":  False,
    "start_date":       datetime(2025, 1, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
}

DRIFT_THRESHOLD = 0.15


def _check_drift(module: str, **context) -> str:
    """
    Run Evidently drift check for a module.
    Pushes drift report to XCom and returns branch task_id.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/../../../")
    from mlops.evidently.drift_monitor import run_supply_chain_drift, run_airline_drift

    if module == "supply_chain":
        report = run_supply_chain_drift(trigger_airflow=False)
    else:
        report = run_airline_drift(trigger_airflow=False)

    psi     = report["summary"]["overall_psi"]
    retrain = report["summary"]["retrain_needed"]
    drifted = report["summary"]["drifted_features"]

    context["task_instance"].xcom_push(key=f"{module}_psi",             value=psi)
    context["task_instance"].xcom_push(key=f"{module}_retrain",         value=retrain)
    context["task_instance"].xcom_push(key=f"{module}_drifted_features", value=drifted)

    print(f"[{module}] PSI={psi:.4f} | Retrain={retrain} | Drifted: {drifted}")

    if retrain:
        return f"retrain_{module}"
    return f"skip_{module}"


def _skip(module: str, **context):
    ti  = context["task_instance"]
    psi = ti.xcom_pull(key=f"{module}_psi")
    print(f"[{module}] Drift PSI={psi:.4f} below threshold {DRIFT_THRESHOLD}. No retrain needed.")


def _notify_drift(module: str, **context):
    """Log drift notification. Extend with Slack/email in production."""
    ti       = context["task_instance"]
    psi      = ti.xcom_pull(key=f"{module}_psi")
    features = ti.xcom_pull(key=f"{module}_drifted_features")
    print(f"[{module}] DRIFT ALERT — PSI={psi:.4f} | Features: {features}")
    print(f"Triggering {module} retraining pipeline...")


with DAG(
    dag_id="weekly_drift_monitoring",
    default_args=default_args,
    description="Weekly Evidently AI drift checks — auto-triggers retraining on drift",
    schedule="0 4 * * 1",   # Every Monday at 4am
    catchup=False,
    tags=["drift", "evidently", "mlops", "weekly"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    #  Supply Chain branch 
    check_sc = BranchPythonOperator(
        task_id="drift_check_supply_chain",
        python_callable=lambda **ctx: _check_drift("supply_chain", **ctx),
    )
    retrain_sc = TriggerDagRunOperator(
        task_id="retrain_supply_chain",
        trigger_dag_id="supply_chain_weekly_pipeline",
        conf={"triggered_by": "drift_monitor", "reason": "psi_exceeded_threshold"},
        wait_for_completion=False,
    )
    notify_sc = PythonOperator(
        task_id="notify_supply_chain_drift",
        python_callable=lambda **ctx: _notify_drift("supply_chain", **ctx),
    )
    skip_sc = PythonOperator(
        task_id="skip_supply_chain",
        python_callable=lambda **ctx: _skip("supply_chain", **ctx),
    )

    # ── Airline branch
    check_al = BranchPythonOperator(
        task_id="drift_check_airline",
        python_callable=lambda **ctx: _check_drift("airline", **ctx),
    )
    retrain_al = TriggerDagRunOperator(
        task_id="retrain_airline",
        trigger_dag_id="airline_weekly_pipeline",
        conf={"triggered_by": "drift_monitor", "reason": "psi_exceeded_threshold"},
        wait_for_completion=False,
    )
    notify_al = PythonOperator(
        task_id="notify_airline_drift",
        python_callable=lambda **ctx: _notify_drift("airline", **ctx),
    )
    skip_al = PythonOperator(
        task_id="skip_airline",
        python_callable=lambda **ctx: _skip("airline", **ctx),
    )

    #  Wire up 
    start >> [check_sc, check_al]

    check_sc >> retrain_sc >> notify_sc >> end
    check_sc >> skip_sc >> end

    check_al >> retrain_al >> notify_al >> end
    check_al >> skip_al >> end
