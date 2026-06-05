"""Tests for the ICD-10/CPT taxonomy graph and compatibility validation."""

from __future__ import annotations

import pandas as pd

from src.taxonomy.compatibility_validator import CompatibilityValidator
from src.taxonomy.taxonomy_graph import TaxonomyGraph
from src.taxonomy.upcoding_detector import UpcodingDetector


def test_graph_known_compatible_pair():
    graph = TaxonomyGraph.from_reference()
    # E11.9 (Type 2 diabetes) with 83036 (HbA1c) is clinically compatible.
    assert graph.is_compatible("E11.9", "83036") is True
    # E11.9 with 93000 (ECG routine) is marked incompatible in the reference.
    assert graph.is_compatible("E11.9", "93000") is False


def test_validator_flags_mismatch_and_ratio():
    df = pd.DataFrame(
        [
            {"icd10_code": "E11.9", "cpt_code": "93000", "allowed_amount": 75,
             "billed_amount": 300, "units": 1},
            {"icd10_code": "E11.9", "cpt_code": "83036", "allowed_amount": 55,
             "billed_amount": 60, "units": 1},
        ]
    )
    out = CompatibilityValidator().validate(df)
    assert out.loc[0, "procedure_diagnosis_mismatch"] == 1
    assert out.loc[1, "procedure_diagnosis_mismatch"] == 0
    assert out.loc[0, "billed_to_allowed_ratio"] == 4.0


def test_upcoding_flag():
    df = pd.DataFrame(
        [{"icd10_code": "E11.9", "cpt_code": "83036", "allowed_amount": 55,
          "billed_amount": 60, "units": 1}]
    )
    out = CompatibilityValidator().validate(df)
    out = UpcodingDetector(upcode_ratio=2.0).detect(out)
    # billed/typical ~ 60/55 -> not upcoded.
    assert out.loc[0, "upcoding_flag"] == 0
