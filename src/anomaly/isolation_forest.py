"""Isolation Forest statistical anomaly detection on billing features.

Trains an Isolation Forest over per-claim billing features and converts the raw
decision-function output into a normalised 0-1 anomaly score, where higher means
more anomalous. Includes a simple threshold optimiser that picks the score cutoff
maximising separation against the contamination prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger("isolation-forest")


class BillingAnomalyDetector:
    def __init__(self, config=CONFIG.anomaly) -> None:
        self.config = config
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.threshold_: float | None = None

    def fit_score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        feat_cols = [c for c in self.config.feature_columns if c in out.columns]
        X = self.scaler_fit_transform(out[feat_cols].fillna(0.0).to_numpy())

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)

        # decision_function: higher = more normal. Invert and min-max to 0-1.
        raw = -self.model.decision_function(X)
        lo, hi = raw.min(), raw.max()
        out["anomaly_score"] = (raw - lo) / (hi - lo) if hi > lo else 0.0
        out["anomaly_flag"] = (self.model.predict(X) == -1).astype(int)

        self.threshold_ = self.optimize_threshold(out["anomaly_score"])
        logger.info(
            "Isolation Forest scored %d claims | %d flagged anomalies | threshold=%.3f",
            len(out), int(out["anomaly_flag"].sum()), self.threshold_,
        )
        return out

    def scaler_fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def optimize_threshold(self, scores: pd.Series) -> float:
        """Pick the threshold at the (1 - contamination) quantile of scores."""
        return float(np.quantile(scores, 1 - self.config.contamination))
