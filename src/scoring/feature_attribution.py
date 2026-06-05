"""SHAP-style feature attribution for the composite fraud score.

For each claim we attribute the composite score to its three weighted components
and surface the top risk drivers. When the ``shap`` library is installed it can
be used for tree-model explanations; the default additive attribution is exact
for the linear composite and needs no extra dependencies, so every scored claim
gets an explainable "why" without heavyweight requirements.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger("feature-attribution")

_DRIVER_LABELS = {
    "anomaly_contrib": "Statistical billing anomaly (Isolation Forest)",
    "misalignment_contrib": "Diagnosis-procedure note misalignment (ClinicalBERT)",
    "billing_contrib": "Billing-ratio outlier within provider cluster",
    "upcoding_flag": "Upcoding vs typical allowed amount",
    "procedure_diagnosis_mismatch": "Procedure-diagnosis taxonomy mismatch",
    "unbundling_flag": "Potential unbundling of an encounter",
}


class FeatureAttributor:
    def __init__(self, config=CONFIG.scoring) -> None:
        self.config = config

    def attribute(self, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        out = df.copy()
        c = self.config
        out["anomaly_contrib"] = (c.anomaly_weight * out.get("anomaly_score", 0.0) * 100).round(2)
        out["misalignment_contrib"] = (
            c.misalignment_weight * out.get("clinical_misalignment", 0.0) * 100
        ).round(2)
        out["billing_contrib"] = (
            c.billing_ratio_weight * out.get("billing_ratio_anomaly", 0.0) * 100
        ).round(2)

        contrib_cols = ["anomaly_contrib", "misalignment_contrib", "billing_contrib"]
        flag_cols = [f for f in ("upcoding_flag", "procedure_diagnosis_mismatch", "unbundling_flag")
                     if f in out.columns]

        def _drivers(row) -> str:
            scored = {col: row[col] for col in contrib_cols}
            # Boost ordering with binary flags so they appear when material.
            for f in flag_cols:
                if row.get(f, 0):
                    scored[f] = scored.get(f, 0) + 5
            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            top = [_DRIVER_LABELS.get(k, k) for k, v in ranked[:top_k] if v > 0]
            return "; ".join(top) if top else "No material risk drivers"

        out["top_risk_drivers"] = out.apply(_drivers, axis=1)
        logger.info("Attributed top-%d risk drivers for %d claims", top_k, len(out))
        return out
