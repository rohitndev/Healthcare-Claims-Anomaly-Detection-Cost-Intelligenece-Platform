"""Cluster profiling and within-cluster billing-ratio anomaly scoring.

Summarises each provider behavioural cluster and computes, for every claim, how
far its billing ratio deviates from the provider's cluster norm. This per-claim
``billing_ratio_anomaly`` (a robust z-score mapped to 0-1) is one of the three
components of the composite fraud index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("cluster-profiler")


class ClusterProfiler:
    def profile(self, provider_features: pd.DataFrame) -> pd.DataFrame:
        agg = (
            provider_features.groupby("provider_cluster")
            .agg(
                providers=("provider_id", "nunique"),
                avg_billed_amount=("avg_billed_amount", "mean"),
                avg_billed_to_allowed_ratio=("avg_billed_to_allowed_ratio", "mean"),
                avg_claims=("claims_count", "mean"),
                avg_mismatch_rate=("mismatch_rate", "mean"),
            )
            .reset_index()
            .sort_values("avg_billed_to_allowed_ratio", ascending=False)
        )
        logger.info("Profiled %d provider clusters", len(agg))
        return agg

    def attach_cluster_anomaly(
        self, claims: pd.DataFrame, provider_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach provider_cluster to each claim and compute within-cluster
        billing-ratio anomaly (0-1)."""
        out = claims.merge(
            provider_features[["provider_id", "provider_cluster"]],
            on="provider_id",
            how="left",
        )
        out["provider_cluster"] = out["provider_cluster"].fillna(-1).astype(int)

        def _robust_anomaly(group: pd.Series) -> pd.Series:
            median = group.median()
            mad = (group - median).abs().median()
            scale = mad if mad > 1e-9 else (group.std() if group.std() > 1e-9 else 1.0)
            z = (group - median).abs() / scale
            return (z / (z + 3.0)).clip(0, 1)  # squashing -> 0-1

        out["billing_ratio_anomaly"] = (
            out.groupby("provider_cluster")["billed_to_allowed_ratio"]
            .transform(_robust_anomaly)
            .fillna(0.0)
        )
        logger.info("Computed within-cluster billing-ratio anomaly for %d claims", len(out))
        return out
