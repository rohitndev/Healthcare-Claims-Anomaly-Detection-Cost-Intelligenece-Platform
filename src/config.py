"""Central configuration for the claims intelligence platform.

All tunable parameters, file paths, scoring weights, and optional cloud settings
live here. Cloud credentials are read from environment variables (see
``.env.example``); when they are absent the platform automatically falls back to
the local filesystem so the pipeline always runs end-to-end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw"
ANONYMIZED_DIR = OUTPUT_DIR / "anonymized"
SCORED_DIR = OUTPUT_DIR / "scored"
AUDIT_DIR = OUTPUT_DIR / "audit"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboards" / "exports"

for _path in (
    OUTPUT_DIR,
    RAW_DIR,
    ANONYMIZED_DIR,
    SCORED_DIR,
    AUDIT_DIR,
    REPORTS_DIR,
    LOG_DIR,
    DASHBOARD_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ScoringConfig:
    """Weights for the composite fraud / waste risk index (0-100)."""

    anomaly_weight: float = 0.40            # Isolation Forest statistical anomaly
    misalignment_weight: float = 0.30       # ClinicalBERT diagnosis-procedure misalignment
    billing_ratio_weight: float = 0.30      # Billing-ratio outlier within provider cluster
    high_risk_threshold: float = 75.0       # Score above which a claim is auto-triaged
    alert_provider_threshold: float = 42.0  # Mean provider score that triggers an SNS alert


@dataclass
class ClusteringConfig:
    """Provider behavioral segmentation settings."""

    n_clusters: int = 8                     # Billing-behaviour segments
    random_state: int = 42
    feature_columns: tuple[str, ...] = (
        "avg_billed_amount",
        "avg_billed_to_allowed_ratio",
        "avg_units",
        "distinct_cpt_count",
        "distinct_icd_count",
        "claims_count",
    )


@dataclass
class AnomalyConfig:
    """Isolation Forest configuration."""

    contamination: float = 0.08
    n_estimators: int = 200
    random_state: int = 42
    feature_columns: tuple[str, ...] = (
        "billed_amount",
        "billed_to_allowed_ratio",
        "units",
        "amount_per_unit",
    )


@dataclass
class CloudConfig:
    """Optional cloud connectivity. Empty values => local fallback."""

    # AWS S3 (HIPAA-compliant claims data lake)
    aws_s3_bucket: str = field(default_factory=lambda: os.getenv("AWS_S3_BUCKET", ""))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))

    # AWS SNS (high-risk claim alerts)
    aws_sns_topic_arn: str = field(default_factory=lambda: os.getenv("AWS_SNS_TOPIC_ARN", ""))

    # Snowflake (fraud scores / audit reports / compliance history)
    snowflake_account: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ACCOUNT", ""))
    snowflake_user: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_USER", ""))
    snowflake_password: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_PASSWORD", ""))
    snowflake_database: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_DATABASE", "CLAIMS_INTELLIGENCE"))
    snowflake_schema: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"))
    snowflake_warehouse: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"))

    @property
    def s3_enabled(self) -> bool:
        return bool(self.aws_s3_bucket)

    @property
    def sns_enabled(self) -> bool:
        return bool(self.aws_sns_topic_arn)

    @property
    def snowflake_enabled(self) -> bool:
        return bool(self.snowflake_account and self.snowflake_user)


@dataclass
class GroqConfig:
    """Groq API (Mixtral) for the clinical audit narrative generator."""

    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class NLPConfig:
    """ClinicalBERT discharge-note analysis settings."""

    model_name: str = field(
        default_factory=lambda: os.getenv("CLINICALBERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT")
    )
    # When transformers/torch are unavailable, a deterministic lexical aligner is used.
    use_transformer: bool = field(default_factory=lambda: _as_bool(os.getenv("USE_CLINICALBERT"), False))
    max_length: int = 256


@dataclass
class PlatformConfig:
    """Top-level configuration object."""

    # Prefer Spark when available; auto-fall back to pandas when it is not.
    use_spark: bool = field(default_factory=lambda: _as_bool(os.getenv("USE_SPARK"), False))
    use_presidio: bool = field(default_factory=lambda: _as_bool(os.getenv("USE_PRESIDIO"), True))
    n_claims: int = field(default_factory=lambda: int(os.getenv("N_CLAIMS", "5000")))
    n_providers: int = field(default_factory=lambda: int(os.getenv("N_PROVIDERS", "120")))
    random_seed: int = 42

    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)


# 18 HIPAA PHI identifiers tracked by the anonymization layer (45 CFR 164.514).
HIPAA_PHI_IDENTIFIERS = (
    "name",
    "geographic_subdivision",
    "dates",
    "phone_number",
    "fax_number",
    "email_address",
    "social_security_number",
    "medical_record_number",
    "health_plan_beneficiary_number",
    "account_number",
    "certificate_license_number",
    "vehicle_identifier",
    "device_identifier",
    "url",
    "ip_address",
    "biometric_identifier",
    "full_face_photo",
    "other_unique_identifier",
)

CONFIG = PlatformConfig()
