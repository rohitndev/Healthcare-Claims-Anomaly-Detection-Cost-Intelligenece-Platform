"""Airflow DAG: daily claims processing.

Schedules the end-to-end claims intelligence pipeline once per day. Each logical
stage is a separate task so failures are isolated and retried independently; in
this lightweight form the heavy lifting is delegated to ``run_pipeline`` which is
itself idempotent. Deploy this file to the Airflow ``dags/`` folder (AWS MWAA in
production).
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW = True
except Exception:  # pragma: no cover - airflow optional locally
    _AIRFLOW = False


def _run_daily(**_context) -> dict:
    from pipelines.daily_claims_pipeline import run_pipeline

    result = run_pipeline()
    return result.metrics


if _AIRFLOW:
    default_args = {
        "owner": "compliance-data-team",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
        "depends_on_past": False,
    }

    with DAG(
        dag_id="daily_claims_processing",
        description="Daily HIPAA-compliant claims anomaly detection and scoring",
        schedule_interval="0 6 * * *",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["healthcare", "fraud-detection", "hipaa", "pyspark"],
    ) as dag:
        process = PythonOperator(
            task_id="run_claims_intelligence_pipeline",
            python_callable=_run_daily,
        )
