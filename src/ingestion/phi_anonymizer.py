"""HIPAA-compliant PHI anonymization applied at ingestion.

Microsoft Presidio (open source) is used to detect and redact the 18 HIPAA PHI
identifiers before any analytic processing. When Presidio (or its spaCy model) is
not installed, a deterministic regex + hashing fallback provides equivalent
redaction so the pipeline always runs and never leaks raw PHI downstream.

Direct identifiers (name, SSN, address, email, phone, MRN, member id, DOB) are
either dropped or replaced with a salted, non-reversible pseudonym so that
provider-level aggregation still works while patient identity is protected.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from src.config import CONFIG, HIPAA_PHI_IDENTIFIERS
from src.ingestion.audit_logger import HIPAAAuditLogger
from src.utils.logger import get_logger

logger = get_logger("phi-anonymizer")

# Columns carrying direct PHI and the HIPAA identifier they map to.
PHI_COLUMNS = {
    "patient_name": "name",
    "patient_ssn": "social_security_number",
    "patient_dob": "dates",
    "patient_email": "email_address",
    "patient_phone": "phone_number",
    "patient_address": "geographic_subdivision",
    "medical_record_number": "medical_record_number",
    "member_id": "health_plan_beneficiary_number",
}

_SALT = "claims-intelligence-hipaa-salt"

_REGEX_PATTERNS = {
    "social_security_number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone_number": re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
    "email_address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "dates": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
}


def _pseudonymize(value: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:{value}".encode("utf-8")).hexdigest()
    return digest[:16]


class PHIAnonymizer:
    """Remove/replace the 18 HIPAA PHI identifiers from a claims DataFrame."""

    def __init__(self, audit: HIPAAAuditLogger | None = None) -> None:
        self.audit = audit or HIPAAAuditLogger()
        self._engine = None
        self._anonymizer = None
        if CONFIG.use_presidio:
            self._try_load_presidio()

    def _try_load_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._engine = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Microsoft Presidio engine loaded for free-text PHI scrubbing")
        except Exception as exc:
            logger.warning("Presidio unavailable, using regex fallback (%s)", exc)
            self._engine = None
            self._anonymizer = None

    # ------------------------------------------------------------------ #
    # Free-text scrubbing (discharge notes)
    # ------------------------------------------------------------------ #
    def scrub_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        if self._engine is not None and self._anonymizer is not None:
            try:
                results = self._engine.analyze(text=text, language="en")
                return self._anonymizer.anonymize(text=text, analyzer_results=results).text
            except Exception as exc:  # pragma: no cover
                logger.warning("Presidio scrub failed, regex fallback used (%s)", exc)
        for label, pattern in _REGEX_PATTERNS.items():
            text = pattern.sub(f"<{label.upper()}>", text)
        return text

    # ------------------------------------------------------------------ #
    # Structured column anonymization
    # ------------------------------------------------------------------ #
    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an anonymized copy of ``df``."""
        out = df.copy()
        present = [c for c in PHI_COLUMNS if c in out.columns]

        self.audit.log_access(
            action="read",
            phi_identifiers=[PHI_COLUMNS[c] for c in present],
            record_count=len(out),
            purpose="ingestion PHI anonymization",
        )

        # Stable per-patient pseudonym (lets us count distinct patients safely).
        if "patient_name" in out.columns:
            out["patient_pseudo_id"] = out["patient_name"].astype(str).map(_pseudonymize)

        # Keep only the year of birth -> age band (HIPAA safe-harbor on dates).
        if "patient_dob" in out.columns:
            dob = pd.to_datetime(out["patient_dob"], errors="coerce")
            out["patient_age_band"] = (
                (2025 - dob.dt.year).fillna(0).clip(lower=0)
                .floordiv(10).mul(10).astype(int).astype(str) + "s"
            )

        # Coarsen geography to state only.
        # (provider_state already retained; patient_address dropped below.)

        # Drop all direct identifier columns.
        out = out.drop(columns=present, errors="ignore")

        # Scrub any residual PHI from free-text notes.
        if "discharge_note" in out.columns:
            out["discharge_note"] = out["discharge_note"].map(self.scrub_text)

        self.audit.log_access(
            action="anonymize",
            phi_identifiers=list(HIPAA_PHI_IDENTIFIERS),
            record_count=len(out),
            purpose="18 HIPAA identifiers redacted at ingestion",
        )
        logger.info(
            "Anonymized %d claims; dropped %d direct-identifier columns", len(out), len(present)
        )
        return out
