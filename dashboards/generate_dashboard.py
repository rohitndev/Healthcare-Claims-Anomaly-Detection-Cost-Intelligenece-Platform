"""Generate compliance dashboard exports.

Produces the visual artefacts the README references and that feed the Power BI
compliance dashboard:

  * provider behavioural cluster scatter
  * fraud-risk score distribution
  * provider risk map (mean score by state)
  * risk-tier composition

It also writes a flat ``powerbi_dataset.csv`` that the Power BI ``.pbix`` connects
to. Charts are saved as PNGs under ``dashboards/exports`` and copied to ``docs/``
so they can be embedded directly in the README.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DASHBOARD_DIR, PROJECT_ROOT, SCORED_DIR
from src.utils.logger import get_logger

logger = get_logger("dashboard")

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

_PRIMARY = "#1f3b6f"
_ACCENT = "#c0392b"
_OK = "#2e7d52"


def _save(fig, name: str) -> Path:
    out = Path(DASHBOARD_DIR) / name
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    shutil.copy(out, DOCS_DIR / name)
    plt.close(fig)
    logger.info("Saved dashboard chart: %s", name)
    return out


def _cluster_scatter(provider_features: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    clusters = sorted(provider_features["provider_cluster"].unique())
    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(clusters):
        sub = provider_features[provider_features["provider_cluster"] == cl]
        ax.scatter(
            sub["avg_billed_amount"], sub["avg_billed_to_allowed_ratio"],
            s=30 + sub["claims_count"], alpha=0.7, color=cmap(i % 10),
            label=f"Cluster {cl}", edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel("Avg billed amount per claim ($)")
    ax.set_ylabel("Avg billed-to-allowed ratio")
    ax.set_title("Provider Behavioural Clusters (K-Means)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    _save(fig, "provider_clusters.png")


def _score_distribution(scored: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scored["fraud_risk_score"], bins=40, color=_PRIMARY, alpha=0.85, edgecolor="white")
    ax.axvline(75, color=_ACCENT, linestyle="--", linewidth=2, label="High-risk threshold (75)")
    ax.set_xlabel("Composite fraud risk score (0-100)")
    ax.set_ylabel("Number of claims")
    ax.set_title("Fraud Risk Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)
    _save(fig, "fraud_score_distribution.png")


def _risk_tier_composition(scored: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ["LOW", "MEDIUM", "HIGH"]
    counts = scored["risk_tier"].value_counts().reindex(order).fillna(0)
    colors = [_OK, "#e0a800", _ACCENT]
    ax.bar(order, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of claims")
    ax.set_title("Claims by Risk Tier")
    ax.grid(True, alpha=0.2, axis="y")
    _save(fig, "risk_tier_composition.png")


def _provider_risk_map(scored: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    by_state = (
        scored.groupby("provider_state")["fraud_risk_score"].mean().sort_values(ascending=False)
        if "provider_state" in scored.columns else pd.Series(dtype=float)
    )
    ax.bar(by_state.index, by_state.values, color=_PRIMARY, edgecolor="white")
    ax.set_ylabel("Mean fraud risk score")
    ax.set_xlabel("Provider state")
    ax.set_title("Provider Risk Map — Mean Fraud Score by State")
    ax.grid(True, alpha=0.2, axis="y")
    _save(fig, "provider_risk_map.png")


def build_dashboards(
    scored: pd.DataFrame | None = None,
    provider_features: pd.DataFrame | None = None,
    cluster_profile: pd.DataFrame | None = None,
) -> None:
    """Build all dashboard exports. Loads scored data from disk if not passed."""
    if scored is None:
        scored = pd.read_parquet(Path(SCORED_DIR) / "claims_scored.parquet")
    if provider_features is None:
        from src.clustering.provider_segmentation import build_provider_features

        provider_features = build_provider_features(scored)
        provider_features["provider_cluster"] = scored.groupby("provider_id")[
            "provider_cluster"
        ].first().reindex(provider_features["provider_id"]).values

    _cluster_scatter(provider_features)
    _score_distribution(scored)
    _risk_tier_composition(scored)
    _provider_risk_map(scored)

    # Power BI flat dataset.
    pbi_path = Path(DASHBOARD_DIR) / "powerbi_dataset.csv"
    scored.to_csv(pbi_path, index=False)
    logger.info("Wrote Power BI dataset -> %s", pbi_path)


if __name__ == "__main__":
    build_dashboards()
