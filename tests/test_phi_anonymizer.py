"""Tests for HIPAA PHI anonymization."""

from __future__ import annotations

import pandas as pd

from src.ingestion.phi_anonymizer import PHI_COLUMNS, PHIAnonymizer


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "CLM1",
                "patient_name": "John Smith",
                "patient_ssn": "123-45-6789",
                "patient_dob": "1980-05-01",
                "patient_email": "john.smith@example.com",
                "patient_phone": "(212) 555-1234",
                "patient_address": "123 Oak St, Springfield, NY",
                "medical_record_number": "MRN1234567",
                "member_id": "MBR123456789",
                "provider_id": "PRV1",
                "provider_state": "NY",
                "discharge_note": "Contact patient at john.smith@example.com or 123-45-6789.",
            }
        ]
    )


def test_direct_identifier_columns_removed():
    out = PHIAnonymizer().anonymize(_sample())
    for col in PHI_COLUMNS:
        assert col not in out.columns


def test_pseudonym_and_age_band_created():
    out = PHIAnonymizer().anonymize(_sample())
    assert "patient_pseudo_id" in out.columns
    assert out["patient_pseudo_id"].iloc[0] != "John Smith"
    assert "patient_age_band" in out.columns


def test_free_text_phi_scrubbed():
    out = PHIAnonymizer().anonymize(_sample())
    note = out["discharge_note"].iloc[0]
    assert "john.smith@example.com" not in note
    assert "123-45-6789" not in note
