"""Tests for composite fraud scoring and feature attribution."""

from __future__ import annotations

import pandas as pd

from src.scoring.composite_score import CompositeScorer
from src.scoring.feature_attribution import FeatureAttributor


def _scored_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"claim_id": "A", "anomaly_score": 0.9, "clinical_misalignment": 0.9,
             "billing_ratio_anomaly": 0.9},
            {"claim_id": "B", "anomaly_score": 0.05, "clinical_misalignment": 0.0,
             "billing_ratio_anomaly": 0.05},
        ]
    )


def test_composite_score_bounds_and_weighting():
    out = CompositeScorer().score(_scored_input())
    assert out.loc[0, "fraud_risk_score"] == 90.0  # 0.9*100
    assert out.loc[0, "high_risk_flag"] == 1
    assert out.loc[1, "high_risk_flag"] == 0
    assert (out["fraud_risk_score"].between(0, 100)).all()


def test_risk_tiers():
    out = CompositeScorer().score(_scored_input())
    assert out.loc[0, "risk_tier"] == "HIGH"
    assert out.loc[1, "risk_tier"] == "LOW"


def test_attribution_lists_drivers():
    scored = CompositeScorer().score(_scored_input())
    out = FeatureAttributor().attribute(scored)
    assert "top_risk_drivers" in out.columns
    assert out.loc[0, "top_risk_drivers"] != "No material risk drivers"
