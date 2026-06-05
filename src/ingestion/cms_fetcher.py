"""Claims data fetcher.

Loads a raw claims dataset from the local data lake (or generates a synthetic
Synthea-style dataset on first run) and hands it to the anonymization stage. In
production this reads CMS Medicare provider-utilization extracts from the S3
landing zone; locally it reads/writes Parquet under ``output/raw``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("cms-fetcher")


class CMSClaimsFetcher:
    """Fetch raw claims into a pandas DataFrame."""

    def __init__(self, raw_path: Path | None = None) -> None:
        self.raw_path = raw_path or (RAW_DIR / "claims_raw.parquet")

    def fetch(self, regenerate: bool = False) -> pd.DataFrame:
        if regenerate or not self.raw_path.exists():
            logger.info("Raw claims not found; generating synthetic dataset")
            from data.generate_claims import generate

            df = generate(CONFIG.n_claims, CONFIG.n_providers, CONFIG.random_seed)
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.raw_path, index=False)
        else:
            logger.info("Loading raw claims from %s", self.raw_path)
            df = pd.read_parquet(self.raw_path)

        logger.info(
            "Fetched %d claims | %d providers", len(df), df["provider_id"].nunique()
        )
        return df
