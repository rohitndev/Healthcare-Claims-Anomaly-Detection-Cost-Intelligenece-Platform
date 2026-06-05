"""Discharge-note analyzer.

Light NLP utilities over the (already anonymized) discharge notes: token/length
statistics and extraction of the clinical terms that the alignment scorer relies
on. Kept separate from the model wrapper so it can be unit-tested without any deep
learning dependencies.
"""

from __future__ import annotations

import re

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("discharge-analyzer")

_TOKEN = re.compile(r"[a-zA-Z]+")


class DischargeNoteAnalyzer:
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        notes = out["discharge_note"].fillna("")
        out["note_token_count"] = notes.map(lambda t: len(_TOKEN.findall(t)))
        out["note_has_redaction"] = notes.str.contains("<", regex=False).astype(int)
        logger.info(
            "Analyzed %d discharge notes (avg %.1f tokens)",
            len(out), out["note_token_count"].mean(),
        )
        return out
