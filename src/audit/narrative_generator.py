"""LLM-powered clinical audit narrative generator (Groq / Mixtral).

For every auto-triaged high-risk claim the Groq API (Mixtral 8x7B) generates a
structured clinical audit report: diagnosis validity, procedure appropriateness,
billing-pattern context, and a recommended action. When ``GROQ_API_KEY`` is not
configured (or the SDK is unavailable), the deterministic template renderer in
``report_templates`` produces the same structured report offline, so the pipeline
always emits an audit narrative for each high-risk claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.audit.report_templates import AuditNarrative, recommended_action, render_template
from src.config import CONFIG, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger("narrative-generator")

_SYSTEM_PROMPT = (
    "You are a clinical claims auditor. Given a healthcare claim's coded diagnosis, "
    "procedure, billing figures, and model risk signals, produce a concise, factual "
    "audit. Respond ONLY with JSON containing the keys: diagnosis_validity, "
    "procedure_appropriateness, billing_pattern_context, recommended_action."
)


class AuditNarrativeGenerator:
    def __init__(self) -> None:
        self.config = CONFIG.groq
        self._client = None
        if self.config.enabled:
            self._try_load_client()

    def _try_load_client(self) -> None:
        try:
            from groq import Groq

            self._client = Groq(api_key=self.config.api_key)
            logger.info("Groq client initialised (model=%s)", self.config.model)
        except Exception as exc:
            logger.warning("Groq unavailable, using template narratives (%s)", exc)
            self._client = None

    def _generate_llm(self, row: dict) -> AuditNarrative | None:  # pragma: no cover - network
        prompt = (
            f"Claim {row.get('claim_id')} | Provider {row.get('provider_id')}\n"
            f"ICD-10: {row.get('icd10_code')} - {row.get('icd10_description')}\n"
            f"CPT: {row.get('cpt_code')} - {row.get('cpt_description')}\n"
            f"Billed: ${row.get('billed_amount')} | Typical allowed: ${row.get('typical_allowed_amount')}\n"
            f"Taxonomy compatible: {not row.get('procedure_diagnosis_mismatch')}\n"
            f"Clinical note misalignment: {row.get('clinical_misalignment'):.2f}\n"
            f"Composite fraud risk score: {row.get('fraud_risk_score')}/100\n"
            f"Top risk drivers: {row.get('top_risk_drivers')}\n"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return AuditNarrative(
                claim_id=str(row.get("claim_id")),
                provider_id=str(row.get("provider_id")),
                fraud_risk_score=float(row.get("fraud_risk_score", 0.0)),
                risk_tier=str(row.get("risk_tier", "")),
                diagnosis_validity=data.get("diagnosis_validity", ""),
                procedure_appropriateness=data.get("procedure_appropriateness", ""),
                billing_pattern_context=data.get("billing_pattern_context", ""),
                recommended_action=data.get("recommended_action")
                or recommended_action(float(row.get("fraud_risk_score", 0.0)),
                                       CONFIG.scoring.high_risk_threshold),
                top_risk_drivers=str(row.get("top_risk_drivers", "")),
                generated_by=f"groq:{self.config.model}",
            )
        except Exception as exc:
            logger.warning("Groq generation failed for %s, template used (%s)",
                           row.get("claim_id"), exc)
            return None

    def generate_for_high_risk(self, df: pd.DataFrame) -> list[dict]:
        """Generate audit narratives for all high-risk claims; persist to JSON."""
        high = df[df["high_risk_flag"] == 1]
        narratives: list[dict] = []
        threshold = CONFIG.scoring.high_risk_threshold

        for _, row in high.iterrows():
            record = row.to_dict()
            narrative = None
            if self._client is not None:
                narrative = self._generate_llm(record)
            if narrative is None:
                narrative = render_template(record, threshold)
            narratives.append(narrative.to_dict())

        out_path = Path(REPORTS_DIR) / "audit_narratives.json"
        out_path.write_text(json.dumps(narratives, indent=2), encoding="utf-8")
        logger.info(
            "Generated %d clinical audit narratives -> %s", len(narratives), out_path
        )
        return narratives
