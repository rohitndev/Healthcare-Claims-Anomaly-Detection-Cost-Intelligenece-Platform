"""Synthetic healthcare claims generator (Synthea-style).

Generates a realistic, fully synthetic CMS-style claims dataset that contains the
18 HIPAA PHI identifiers (so the anonymization stage has something to remove), a
provider population with distinct billing behaviours, ICD-10/CPT coded line items,
and free-text discharge notes. A configurable fraction of claims are seeded with
fraud/waste patterns (upcoding, procedure-diagnosis mismatch, billing outliers,
and clinically-misaligned notes) so the detection layers have signal to find.

No real patient data is ever used. Run directly::

    python -m data.generate_claims --n-claims 5000 --n-providers 120
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, RAW_DIR, REFERENCE_DIR

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Maria", "Susan", "Jose", "Karen",
    "Aisha", "Wei", "Ravi", "Sofia", "Omar", "Mei", "Carlos", "Fatima",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Patel", "Nguyen", "Kim", "Chen", "Okafor", "Singh", "Ali", "Khan", "Park",
]
CITIES = [
    "Springfield", "Riverside", "Franklin", "Greenville", "Bristol", "Clinton",
    "Fairview", "Salem", "Madison", "Georgetown", "Arlington", "Ashland",
]
STREETS = ["Maple", "Oak", "Cedar", "Pine", "Elm", "Washington", "Lake", "Hill"]
SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Cardiology", "Orthopedics",
    "Pulmonology", "Gastroenterology", "OB/GYN", "Psychiatry",
]
PLACES_OF_SERVICE = ["Office", "Inpatient Hospital", "Outpatient Hospital", "Emergency Room"]


def _load_taxonomy() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "icd10_cpt_compatibility.csv")


def _fake_ssn(rng: random.Random) -> str:
    return f"{rng.randint(100, 899):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"


def _fake_phone(rng: random.Random) -> str:
    return f"({rng.randint(200, 999):03d}) {rng.randint(200, 999):03d}-{rng.randint(1000, 9999):04d}"


def _build_discharge_note(rng: random.Random, row: pd.Series, aligned: bool) -> str:
    """Compose a short discharge note. When ``aligned`` is False the narrative
    intentionally describes an unrelated clinical presentation."""
    keywords = str(row["note_keywords"]).split(";")
    if aligned:
        focus = ", ".join(rng.sample(keywords, k=min(2, len(keywords))))
        return (
            f"Patient presented with {focus}. Examination consistent with "
            f"{row['icd10_description'].lower()}. {row['cpt_description']} performed. "
            f"Condition stable at discharge; follow-up advised."
        )
    # Misaligned: borrow keywords from a different, incompatible condition.
    distractor = rng.choice(
        ["seasonal allergic rhinitis", "minor ankle sprain", "routine vaccination",
         "mild tension headache", "dental caries follow-up"]
    )
    return (
        f"Patient seen for {distractor}. No acute distress noted. Vitals within "
        f"normal limits. Patient discharged with routine care instructions."
    )


def generate(
    n_claims: int = CONFIG.n_claims,
    n_providers: int = CONFIG.n_providers,
    seed: int = CONFIG.random_seed,
    fraud_rate: float = 0.12,
) -> pd.DataFrame:
    """Generate a synthetic claims DataFrame."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    taxonomy = _load_taxonomy()

    # Build providers, a handful of which are "abusive" billers.
    providers = []
    for i in range(n_providers):
        abusive = rng.random() < 0.10
        providers.append(
            {
                "provider_id": f"PRV{i:05d}",
                "provider_npi": f"{rng.randint(1000000000, 1999999999)}",
                "provider_specialty": rng.choice(SPECIALTIES),
                "provider_state": rng.choice(["CA", "TX", "NY", "FL", "IL", "PA"]),
                "abusive": abusive,
            }
        )

    rows = []
    for c in range(n_claims):
        prov = rng.choice(providers)
        tax = taxonomy.iloc[rng.randrange(len(taxonomy))]

        is_fraud = (rng.random() < (fraud_rate * (3.0 if prov["abusive"] else 1.0)))

        # Compatibility: fraudulent claims may pair an incompatible CPT.
        compatible = int(tax["compatible"])
        if is_fraud and rng.random() < 0.5:
            wrong = taxonomy[taxonomy["icd10_code"] != tax["icd10_code"]].sample(
                1, random_state=int(np_rng.integers(0, 1_000_000))
            ).iloc[0]
            cpt_code = wrong["cpt_code"]
            cpt_description = wrong["cpt_description"]
            typical = float(wrong["typical_allowed_amount"])
            compatible = 0
        else:
            cpt_code = tax["cpt_code"]
            cpt_description = tax["cpt_description"]
            typical = float(tax["typical_allowed_amount"])

        units = rng.randint(1, 3)
        allowed_amount = round(typical * units * np_rng.uniform(0.9, 1.1), 2)

        # Upcoding / billing outliers: inflate billed amount well beyond allowed.
        if is_fraud:
            billed_multiplier = np_rng.uniform(2.5, 6.0)
            note_aligned = rng.random() < 0.3
        else:
            billed_multiplier = np_rng.uniform(1.0, 1.4)
            note_aligned = rng.random() < 0.95

        billed_amount = round(allowed_amount * billed_multiplier, 2)

        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        claim_year = rng.choice([2024, 2025])
        claim_date = (
            pd.Timestamp(f"{claim_year}-01-01")
            + pd.Timedelta(days=int(np_rng.integers(0, 364)))
        ).date()
        dob = (
            pd.Timestamp("1945-01-01")
            + pd.Timedelta(days=int(np_rng.integers(0, 26000)))
        ).date()

        row = pd.Series(
            {
                "icd10_code": tax["icd10_code"],
                "icd10_description": tax["icd10_description"],
                "cpt_description": cpt_description,
                "note_keywords": tax["note_keywords"],
            }
        )
        discharge_note = _build_discharge_note(rng, row, note_aligned)

        rows.append(
            {
                # ---- PHI (to be anonymized at ingestion) ----
                "claim_id": f"CLM{c:08d}",
                "patient_name": f"{first} {last}",
                "patient_ssn": _fake_ssn(rng),
                "patient_dob": dob.isoformat(),
                "patient_email": f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com",
                "patient_phone": _fake_phone(rng),
                "patient_address": f"{rng.randint(100, 9999)} {rng.choice(STREETS)} St, "
                                   f"{rng.choice(CITIES)}, {prov['provider_state']}",
                "medical_record_number": f"MRN{rng.randint(1000000, 9999999)}",
                "member_id": f"MBR{rng.randint(100000000, 999999999)}",
                # ---- Provider ----
                "provider_id": prov["provider_id"],
                "provider_npi": prov["provider_npi"],
                "provider_specialty": prov["provider_specialty"],
                "provider_state": prov["provider_state"],
                # ---- Claim line ----
                "claim_date": claim_date.isoformat(),
                "icd10_code": tax["icd10_code"],
                "icd10_description": tax["icd10_description"],
                "cpt_code": cpt_code,
                "cpt_description": cpt_description,
                "place_of_service": rng.choice(PLACES_OF_SERVICE),
                "units": units,
                "allowed_amount": allowed_amount,
                "billed_amount": billed_amount,
                "discharge_note": discharge_note,
                # ---- Ground-truth label (for evaluation only; not used by models) ----
                "is_fraud_label": int(is_fraud),
                "taxonomy_compatible_label": compatible,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> Path:
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare claims.")
    parser.add_argument("--n-claims", type=int, default=CONFIG.n_claims)
    parser.add_argument("--n-providers", type=int, default=CONFIG.n_providers)
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "claims_raw.parquet"))
    args = parser.parse_args()

    df = generate(args.n_claims, args.n_providers, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Generated {len(df):,} claims for {df['provider_id'].nunique()} providers")
    print(f"  Fraud-labelled claims : {int(df['is_fraud_label'].sum()):,}")
    print(f"  Written to            : {out_path}")
    return out_path


if __name__ == "__main__":
    main()
