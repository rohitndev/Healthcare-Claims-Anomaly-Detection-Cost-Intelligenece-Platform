"""End-to-end daily claims processing pipeline.

Orchestrates the full flow in dependency order:

    fetch -> anonymize (Presidio) -> taxonomy validate -> upcoding detect
          -> provider clustering -> within-cluster billing anomaly
          -> Isolation Forest -> ClinicalBERT alignment -> composite score
          -> SHAP-style attribution -> warehouse load -> audit narratives -> SNS alerts

The same callable is invoked by the Airflow ``daily_claims_dag`` in production and
by ``run_pipeline.py`` for a single local run. Every stage degrades gracefully so
the pipeline completes whether or not Spark / Presidio / ClinicalBERT / Groq / AWS
/ Snowflake are configured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.anomaly.isolation_forest import BillingAnomalyDetector
from src.audit.narrative_generator import AuditNarrativeGenerator
from src.audit.sns_alerter import SNSAlerter
from src.clustering.cluster_profiler import ClusterProfiler
from src.clustering.provider_segmentation import ProviderSegmentation
from src.config import ANONYMIZED_DIR, CONFIG, REPORTS_DIR, SCORED_DIR
from src.ingestion.audit_logger import HIPAAAuditLogger
from src.ingestion.cms_fetcher import CMSClaimsFetcher
from src.ingestion.phi_anonymizer import PHIAnonymizer
from src.nlp.clinical_bert import ClinicalBERTAligner
from src.nlp.discharge_analyzer import DischargeNoteAnalyzer
from src.scoring.composite_score import CompositeScorer
from src.scoring.feature_attribution import FeatureAttributor
from src.storage.datalake import DataLakeWriter
from src.storage.warehouse import WarehouseLoader
from src.taxonomy.compatibility_validator import CompatibilityValidator
from src.taxonomy.taxonomy_graph import TaxonomyGraph
from src.taxonomy.upcoding_detector import UpcodingDetector
from src.utils.logger import get_logger
from src.utils.spark_session import engine_name

logger = get_logger("pipeline")

SCORED_COLUMNS = [
    "claim_id", "provider_id", "provider_specialty", "provider_state", "claim_date",
    "icd10_code", "icd10_description", "cpt_code", "cpt_description",
    "units", "allowed_amount", "billed_amount", "typical_allowed_amount",
    "billed_to_allowed_ratio", "billed_to_typical_ratio",
    "provider_cluster", "anomaly_score", "anomaly_flag", "clinical_misalignment",
    "billing_ratio_anomaly", "procedure_diagnosis_mismatch", "upcoding_flag",
    "unbundling_flag", "fraud_risk_score", "risk_tier", "high_risk_flag",
    "top_risk_drivers", "is_fraud_label",
]


@dataclass
class PipelineResult:
    scored: pd.DataFrame
    provider_features: pd.DataFrame
    cluster_profile: pd.DataFrame
    narratives: list
    alerts: list
    metrics: dict


def run_pipeline(regenerate: bool = False) -> PipelineResult:
    start = time.time()
    logger.info("=" * 78)
    logger.info("Healthcare Claims Anomaly Detection & Cost Intelligence Platform")
    logger.info("Execution engine: %s", engine_name())
    logger.info("=" * 78)

    # 1. Ingest -----------------------------------------------------------
    audit = HIPAAAuditLogger()
    claims = CMSClaimsFetcher().fetch(regenerate=regenerate)

    # 2. HIPAA PHI anonymization -----------------------------------------
    anonymizer = PHIAnonymizer(audit=audit)
    claims = anonymizer.anonymize(claims)
    DataLakeWriter().write_parquet(
        claims, Path(ANONYMIZED_DIR) / "claims_anonymized.parquet",
        s3_key="anonymized/claims_anonymized.parquet",
    )

    # 3. Taxonomy validation + upcoding ----------------------------------
    graph = TaxonomyGraph.from_reference()
    claims = CompatibilityValidator(graph).validate(claims)
    claims = UpcodingDetector().detect(claims)

    # 4. Provider clustering ---------------------------------------------
    segmentation = ProviderSegmentation()
    provider_features = segmentation.fit_predict(claims)
    profiler = ClusterProfiler()
    cluster_profile = profiler.profile(provider_features)
    claims = profiler.attach_cluster_anomaly(claims, provider_features)

    # 5. Isolation Forest anomaly ----------------------------------------
    claims = BillingAnomalyDetector().fit_score(claims)

    # 6. ClinicalBERT diagnosis-procedure alignment ----------------------
    claims = DischargeNoteAnalyzer().analyze(claims)
    claims = ClinicalBERTAligner(graph).score(claims)

    # 7. Composite scoring + attribution ---------------------------------
    claims = CompositeScorer().score(claims)
    claims = FeatureAttributor().attribute(claims)

    # 8. Persist curated outputs (Snowflake / local) ---------------------
    scored = claims[[c for c in SCORED_COLUMNS if c in claims.columns]].copy()
    warehouse = WarehouseLoader()
    warehouse.load_table(scored, "fraud_scores")
    warehouse.load_table(cluster_profile, "provider_cluster_profile")
    provider_risk = (
        scored.groupby("provider_id")
        .agg(mean_fraud_score=("fraud_risk_score", "mean"),
             high_risk_claims=("high_risk_flag", "sum"),
             claims=("claim_id", "count"))
        .reset_index()
        .sort_values("mean_fraud_score", ascending=False)
    )
    warehouse.load_table(provider_risk, "provider_risk_summary")
    warehouse.close()
    scored.to_parquet(Path(SCORED_DIR) / "claims_scored.parquet", index=False)

    # 9. LLM audit narratives for high-risk claims -----------------------
    narratives = AuditNarrativeGenerator().generate_for_high_risk(claims)

    # 10. SNS alerts for high-risk providers -----------------------------
    alerts = SNSAlerter().alert_high_risk_providers(scored)

    # Metrics -------------------------------------------------------------
    elapsed = time.time() - start
    metrics = _compute_metrics(scored, provider_features, narratives, alerts, elapsed)
    Path(REPORTS_DIR, "pipeline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _log_summary(metrics)

    return PipelineResult(scored, provider_features, cluster_profile, narratives, alerts, metrics)


def _compute_metrics(scored, provider_features, narratives, alerts, elapsed) -> dict:
    total = len(scored)
    high = int(scored["high_risk_flag"].sum())
    metrics = {
        "engine": engine_name(),
        "runtime_seconds": round(elapsed, 2),
        "total_claims": total,
        "total_providers": int(scored["provider_id"].nunique()),
        "provider_clusters": int(provider_features["provider_cluster"].nunique()),
        "high_risk_claims": high,
        "high_risk_rate_pct": round(100 * high / max(total, 1), 2),
        "audit_narratives_generated": len(narratives),
        "providers_alerted": len(alerts),
        "mean_fraud_score": round(float(scored["fraud_risk_score"].mean()), 2),
        "total_billed_amount": round(float(scored["billed_amount"].sum()), 2),
        "high_risk_billed_amount": round(
            float(scored.loc[scored["high_risk_flag"] == 1, "billed_amount"].sum()), 2
        ),
    }
    # Detection quality vs the synthetic ground-truth label (evaluation only).
    if "is_fraud_label" in scored.columns:
        tp = int(((scored["high_risk_flag"] == 1) & (scored["is_fraud_label"] == 1)).sum())
        fp = int(((scored["high_risk_flag"] == 1) & (scored["is_fraud_label"] == 0)).sum())
        fn = int(((scored["high_risk_flag"] == 0) & (scored["is_fraud_label"] == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics["detection_precision"] = round(precision, 3)
        metrics["detection_recall"] = round(recall, 3)
        metrics["detection_f1"] = round(
            2 * precision * recall / max(precision + recall, 1e-9), 3
        )
    return metrics


def _log_summary(metrics: dict) -> None:
    logger.info("-" * 78)
    logger.info("PIPELINE SUMMARY")
    for key, value in metrics.items():
        logger.info("  %-28s : %s", key, value)
    logger.info("-" * 78)


if __name__ == "__main__":
    run_pipeline()
