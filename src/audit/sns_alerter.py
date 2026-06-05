"""High-risk claim alerting via AWS SNS.

When a provider's mean fraud risk score (or any individual claim) crosses the
configured threshold, an alert is published to an AWS SNS topic so the compliance
team is notified in near-real-time. When SNS is not configured the alert payloads
are written to a local JSONL sink instead, so alerting behaviour is observable
without any cloud account.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import CONFIG, AUDIT_DIR
from src.utils.logger import get_logger

logger = get_logger("sns-alerter")


class SNSAlerter:
    def __init__(self) -> None:
        self.cloud = CONFIG.cloud
        self._client = None
        self.local_sink = Path(AUDIT_DIR) / "alerts.jsonl"
        if self.cloud.sns_enabled:
            self._try_load_client()

    def _try_load_client(self) -> None:
        try:
            import boto3

            self._client = boto3.client("sns", region_name=self.cloud.aws_region)
            logger.info("AWS SNS client initialised (region=%s)", self.cloud.aws_region)
        except Exception as exc:
            logger.warning("SNS unavailable, alerts will be written locally (%s)", exc)
            self._client = None

    def _publish(self, subject: str, message: dict) -> None:
        payload = json.dumps(message, indent=2, default=str)
        if self._client is not None:  # pragma: no cover - network
            try:
                self._client.publish(
                    TopicArn=self.cloud.aws_sns_topic_arn, Subject=subject[:100], Message=payload
                )
                logger.info("Published SNS alert: %s", subject)
                return
            except Exception as exc:
                logger.warning("SNS publish failed, writing locally (%s)", exc)
        with self.local_sink.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"subject": subject, "message": message,
                                     "ts": datetime.now(timezone.utc).isoformat()},
                                    default=str) + "\n")

    def alert_high_risk_providers(self, scored: pd.DataFrame) -> list[dict]:
        """Emit one alert per provider whose mean score exceeds the threshold."""
        thr = CONFIG.scoring.alert_provider_threshold
        provider_risk = (
            scored.groupby("provider_id")
            .agg(mean_score=("fraud_risk_score", "mean"),
                 high_risk_claims=("high_risk_flag", "sum"),
                 claims=("claim_id", "count"))
            .reset_index()
        )
        flagged = provider_risk[provider_risk["mean_score"] >= thr]
        alerts = []
        for _, r in flagged.iterrows():
            msg = {
                "provider_id": r["provider_id"],
                "mean_fraud_score": round(float(r["mean_score"]), 2),
                "high_risk_claims": int(r["high_risk_claims"]),
                "total_claims": int(r["claims"]),
                "threshold": thr,
            }
            self._publish(f"HIGH-RISK PROVIDER {r['provider_id']} score {r['mean_score']:.1f}", msg)
            alerts.append(msg)

        logger.info(
            "Provider risk alerting complete | %d providers above threshold %.0f (sink=%s)",
            len(alerts), thr, "SNS" if self._client is not None else "local",
        )
        return alerts
