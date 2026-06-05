"""Helper for sourcing real public claims data.

In production the platform ingests CMS Medicare Provider Utilization data
(data.cms.gov, free), MIMIC-III clinical notes (PhysioNet, free with
credentialing), and Synthea synthetic patient records (synthea.mitre.org).

These sources require account registration / data-use agreements, so this module
documents the endpoints and downloads the open CMS provider utilization extract
when a URL is supplied. For a zero-setup run, use ``data/generate_claims.py``
instead, which produces an equivalent synthetic dataset.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from src.config import RAW_DIR

CMS_SOURCES = {
    "cms_provider_utilization": "https://data.cms.gov/provider-summary-by-type-of-service",
    "mimic_iii_notes": "https://physionet.org/content/mimiciii/",
    "synthea": "https://synthea.mitre.org/downloads",
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - user-supplied trusted URL
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public CMS claims data.")
    parser.add_argument("--url", help="Direct CSV/Parquet URL of a CMS extract.")
    parser.add_argument("--out", default=str(RAW_DIR / "cms_provider_utilization.csv"))
    args = parser.parse_args()

    if not args.url:
        print("No --url supplied. Public source references:")
        for name, ref in CMS_SOURCES.items():
            print(f"  - {name}: {ref}")
        print("\nFor a self-contained run use: python -m data.generate_claims")
        return

    download(args.url, Path(args.out))


if __name__ == "__main__":
    main()
