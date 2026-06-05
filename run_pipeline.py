"""Command-line entrypoint for the claims intelligence platform.

Examples::

    # Full run (generates synthetic data on first run, scores, audits, alerts)
    python run_pipeline.py

    # Regenerate the synthetic dataset, then run, then build dashboards
    python run_pipeline.py --regenerate --dashboards
"""

from __future__ import annotations

import argparse

from pipelines.daily_claims_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Healthcare Claims Anomaly Detection & Cost Intelligence Platform"
    )
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate the synthetic claims dataset before running.")
    parser.add_argument("--dashboards", action="store_true",
                        help="Build dashboard chart exports after scoring.")
    args = parser.parse_args()

    result = run_pipeline(regenerate=args.regenerate)

    if args.dashboards:
        from dashboards.generate_dashboard import build_dashboards

        build_dashboards(result.scored, result.provider_features, result.cluster_profile)

    print("\nTop 5 highest-risk claims:")
    cols = ["claim_id", "provider_id", "icd10_code", "cpt_code",
            "billed_amount", "fraud_risk_score", "risk_tier"]
    top = result.scored.sort_values("fraud_risk_score", ascending=False).head(5)
    print(top[[c for c in cols if c in top.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
