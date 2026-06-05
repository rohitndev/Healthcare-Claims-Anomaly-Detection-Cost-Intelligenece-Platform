"""Airflow DAG: quarterly model retraining.

Retrains the K-Means provider segmentation and Isolation Forest anomaly models on
the most recent labelled fraud cases, registers the new versions (MLflow in
production), and regenerates the compliance dashboard exports. Runs on the first
day of each quarter.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW = True
except Exception:  # pragma: no cover - airflow optional locally
    _AIRFLOW = False


def _retrain(**_context) -> dict:
    # Regenerate features + models against the latest data and refresh dashboards.
    from dashboards.generate_dashboard import build_dashboards
    from pipelines.daily_claims_pipeline import run_pipeline

    result = run_pipeline(regenerate=True)
    build_dashboards(result.scored, result.provider_features, result.cluster_profile)
    return result.metrics


if _AIRFLOW:
    default_args = {
        "owner": "ml-platform-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=30),
    }

    with DAG(
        dag_id="quarterly_model_retrain",
        description="Quarterly retraining of clustering + anomaly models",
        schedule_interval="0 2 1 */3 *",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["healthcare", "mlops", "retrain"],
    ) as dag:
        retrain = PythonOperator(
            task_id="retrain_models_and_refresh_dashboards",
            python_callable=_retrain,
        )
