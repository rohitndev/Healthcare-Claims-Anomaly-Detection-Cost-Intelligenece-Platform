"""Diagnosis-procedure compatibility validation at scale.

Joins each claim against the taxonomy graph to flag procedure-diagnosis
mismatches (a CPT billed against an ICD-10 it is not clinically compatible with)
and computes the billed-to-allowed ratio used downstream by the anomaly and
scoring layers.
"""

from __future__ import annotations

import pandas as pd

from src.taxonomy.taxonomy_graph import TaxonomyGraph
from src.utils.logger import get_logger

logger = get_logger("compatibility-validator")


class CompatibilityValidator:
    def __init__(self, graph: TaxonomyGraph | None = None) -> None:
        self.graph = graph or TaxonomyGraph.from_reference()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["taxonomy_compatible"] = [
            int(self.graph.is_compatible(icd, cpt))
            for icd, cpt in zip(out["icd10_code"], out["cpt_code"])
        ]
        out["procedure_diagnosis_mismatch"] = 1 - out["taxonomy_compatible"]

        typical = [
            self.graph.typical_allowed(icd, cpt)
            for icd, cpt in zip(out["icd10_code"], out["cpt_code"])
        ]
        out["typical_allowed_amount"] = [
            t if t is not None else a
            for t, a in zip(typical, out["allowed_amount"])
        ]

        allowed = out["allowed_amount"].replace(0, pd.NA)
        out["billed_to_allowed_ratio"] = (out["billed_amount"] / allowed).fillna(1.0)
        out["amount_per_unit"] = out["billed_amount"] / out["units"].clip(lower=1)

        n_mismatch = int(out["procedure_diagnosis_mismatch"].sum())
        logger.info(
            "Validated %d claims | %d procedure-diagnosis mismatches (%.1f%%)",
            len(out), n_mismatch, 100 * n_mismatch / max(len(out), 1),
        )
        return out
