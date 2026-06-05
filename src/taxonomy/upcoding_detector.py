"""Upcoding and unbundling detection.

Upcoding is billing a more expensive service than was rendered. We approximate it
by comparing each claim's billed amount against the taxonomy's typical allowed
amount for that diagnosis-procedure pair, flagging claims whose billed amount is a
configurable multiple above the norm. Unbundling is approximated by detecting
provider/patient/day combinations that split a single encounter into many
separately-billed line items.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("upcoding-detector")


class UpcodingDetector:
    def __init__(self, upcode_ratio: float = 2.0, unbundle_threshold: int = 4) -> None:
        self.upcode_ratio = upcode_ratio
        self.unbundle_threshold = unbundle_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        typical = out["typical_allowed_amount"].replace(0, pd.NA)
        out["billed_to_typical_ratio"] = (out["billed_amount"] / typical).fillna(1.0)
        out["upcoding_flag"] = (out["billed_to_typical_ratio"] >= self.upcode_ratio).astype(int)

        # Unbundling: many distinct CPTs for the same patient/provider/day.
        group_cols = [c for c in ("provider_id", "patient_pseudo_id", "claim_date") if c in out.columns]
        if len(group_cols) == 3:
            line_counts = (
                out.groupby(group_cols)["cpt_code"].transform("nunique")
            )
            out["unbundling_flag"] = (line_counts >= self.unbundle_threshold).astype(int)
        else:
            out["unbundling_flag"] = 0

        logger.info(
            "Upcoding flags: %d | Unbundling flags: %d",
            int(out["upcoding_flag"].sum()), int(out["unbundling_flag"].sum()),
        )
        return out
