"""ICD-10/CPT procedure-diagnosis compatibility graph.

Builds an in-memory compatibility graph from the reference taxonomy and exposes
fast lookups for diagnosis-procedure validity and typical allowed amounts. In a
PySpark deployment this graph is published as a broadcast variable so taxonomy
joins run without shuffling the reference table to every executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import REFERENCE_DIR
from src.utils.logger import get_logger

logger = get_logger("taxonomy-graph")


@dataclass
class TaxonomyGraph:
    """Procedure-diagnosis compatibility graph."""

    _edges: dict[tuple[str, str], bool] = field(default_factory=dict)
    _typical_allowed: dict[tuple[str, str], float] = field(default_factory=dict)
    _note_keywords: dict[str, list[str]] = field(default_factory=dict)
    _icd_descriptions: dict[str, str] = field(default_factory=dict)
    _icd_valid_cpts: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_reference(cls, path=None) -> "TaxonomyGraph":
        path = path or (REFERENCE_DIR / "icd10_cpt_compatibility.csv")
        df = pd.read_csv(path)
        graph = cls()
        for _, r in df.iterrows():
            key = (r["icd10_code"], str(r["cpt_code"]))
            graph._edges[key] = bool(int(r["compatible"]))
            graph._typical_allowed[key] = float(r["typical_allowed_amount"])
            graph._icd_descriptions[r["icd10_code"]] = r["icd10_description"]
            kws = [k.strip().lower() for k in str(r["note_keywords"]).split(";") if k.strip()]
            graph._note_keywords[r["icd10_code"]] = kws
            if bool(int(r["compatible"])):
                graph._icd_valid_cpts.setdefault(r["icd10_code"], set()).add(str(r["cpt_code"]))
        logger.info(
            "Built taxonomy graph: %d edges, %d diagnoses", len(graph._edges), len(graph._icd_valid_cpts)
        )
        return graph

    # -- lookups ---------------------------------------------------------- #
    def is_compatible(self, icd10: str, cpt: str) -> bool:
        key = (icd10, str(cpt))
        if key in self._edges:
            return self._edges[key]
        # Unknown pairing for a known diagnosis => treat as incompatible.
        return False if icd10 in self._icd_valid_cpts else True

    def typical_allowed(self, icd10: str, cpt: str) -> float | None:
        return self._typical_allowed.get((icd10, str(cpt)))

    def note_keywords(self, icd10: str) -> list[str]:
        return self._note_keywords.get(icd10, [])

    def describe(self, icd10: str) -> str:
        return self._icd_descriptions.get(icd10, "")
