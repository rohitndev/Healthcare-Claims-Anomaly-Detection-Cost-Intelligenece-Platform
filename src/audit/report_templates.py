"""Structured clinical audit report templates.

Defines the schema of the audit narrative and a deterministic template renderer
used when the Groq LLM is not configured, so every high-risk claim still receives
a structured, human-readable audit report covering diagnosis validity, procedure
appropriateness, billing-pattern context, and a recommended action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AuditNarrative:
    claim_id: str
    provider_id: str
    fraud_risk_score: float
    risk_tier: str
    diagnosis_validity: str
    procedure_appropriateness: str
    billing_pattern_context: str
    recommended_action: str
    top_risk_drivers: str
    generated_by: str

    def to_dict(self) -> dict:
        return asdict(self)


def recommended_action(score: float, threshold: float) -> str:
    if score >= 90:
        return "Refer to Special Investigations Unit (SIU); suspend payment pending review."
    if score >= threshold:
        return "Route to compliance analyst for manual claim review before adjudication."
    if score >= 40:
        return "Flag for periodic provider trend monitoring; no immediate action."
    return "No action; claim within expected billing norms."


def render_template(row: dict, threshold: float) -> AuditNarrative:
    """Render a structured audit narrative without an LLM (fallback path)."""
    mismatch = int(row.get("procedure_diagnosis_mismatch", 0))
    upcode = int(row.get("upcoding_flag", 0))
    misalign = float(row.get("clinical_misalignment", 0.0))

    diagnosis_validity = (
        f"Discharge note alignment with ICD-10 {row.get('icd10_code')} "
        f"({row.get('icd10_description', 'n/a')}) is "
        + ("WEAK; documented presentation does not clearly support the coded diagnosis."
           if misalign >= 0.5 else "consistent with the coded diagnosis.")
    )
    procedure_appropriateness = (
        f"CPT {row.get('cpt_code')} ({row.get('cpt_description', 'n/a')}) is "
        + ("NOT clinically compatible with the billed diagnosis (taxonomy mismatch)."
           if mismatch else "compatible with the billed diagnosis.")
    )
    billing_pattern_context = (
        f"Billed ${row.get('billed_amount', 0):,.2f} vs typical allowed "
        f"${row.get('typical_allowed_amount', 0):,.2f} "
        f"(ratio {row.get('billed_to_typical_ratio', 1):.1f}x). "
        + ("Pattern indicates probable upcoding." if upcode else "Within expected range.")
    )

    return AuditNarrative(
        claim_id=str(row.get("claim_id")),
        provider_id=str(row.get("provider_id")),
        fraud_risk_score=float(row.get("fraud_risk_score", 0.0)),
        risk_tier=str(row.get("risk_tier", "")),
        diagnosis_validity=diagnosis_validity,
        procedure_appropriateness=procedure_appropriateness,
        billing_pattern_context=billing_pattern_context,
        recommended_action=recommended_action(float(row.get("fraud_risk_score", 0.0)), threshold),
        top_risk_drivers=str(row.get("top_risk_drivers", "")),
        generated_by="template",
    )
