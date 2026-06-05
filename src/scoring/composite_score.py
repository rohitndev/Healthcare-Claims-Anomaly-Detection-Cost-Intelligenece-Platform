"""Composite fraud / waste risk index (0-100).

Federated scoring combines three independent signals into a single per-claim
risk index::

    composite = 100 * ( w_anomaly       * isolation_forest_score
                      + w_misalignment  * clinical_misalignment
                      + w_billing_ratio * billing_ratio_anomaly )

with weights 0.40 / 0.30 / 0.30 by default. Claims scoring above the high-risk
threshold (75) are auto-triaged for human review and an LLM audit narrative.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger("composite-score")


class CompositeScorer:
    def __init__(self, config=CONFIG.scoring) -> None:
        self.config = config

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c = self.config

        anomaly = out.get("anomaly_score", 0.0)
        misalign = out.get("clinical_misalignment", 0.0)
        billing = out.get("billing_ratio_anomaly", 0.0)

        composite = (
            c.anomaly_weight * anomaly
            + c.misalignment_weight * misalign
            + c.billing_ratio_weight * billing
        )
        out["fraud_risk_score"] = (composite * 100).clip(0, 100).round(2)

        out["risk_tier"] = pd.cut(
            out["fraud_risk_score"],
            bins=[-0.1, 40, c.high_risk_threshold, 100],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)
        out["high_risk_flag"] = (out["fraud_risk_score"] > c.high_risk_threshold).astype(int)

        n_high = int(out["high_risk_flag"].sum())
        logger.info(
            "Composite scoring complete | %d HIGH-risk claims (>%.0f) of %d (%.1f%%)",
            n_high, c.high_risk_threshold, len(out), 100 * n_high / max(len(out), 1),
        )
        return out
