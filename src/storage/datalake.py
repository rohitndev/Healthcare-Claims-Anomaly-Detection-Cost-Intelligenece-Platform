"""HIPAA-compliant claims data lake writer (AWS S3 + Apache Parquet).

Writes anonymized claims and scored outputs as Parquet. When an S3 bucket is
configured (``AWS_S3_BUCKET``) the partitions are uploaded to S3; otherwise they
are written to the local ``output/`` tree. Either way the on-disk/object layout is
identical, so promoting from local to cloud is a configuration change only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger("datalake")


class DataLakeWriter:
    def __init__(self) -> None:
        self.cloud = CONFIG.cloud
        self._s3 = None
        if self.cloud.s3_enabled:
            self._try_load_s3()

    def _try_load_s3(self) -> None:
        try:
            import boto3

            self._s3 = boto3.client("s3", region_name=self.cloud.aws_region)
            logger.info("S3 data lake enabled: s3://%s", self.cloud.aws_s3_bucket)
        except Exception as exc:
            logger.warning("S3 unavailable, writing Parquet locally (%s)", exc)
            self._s3 = None

    def write_parquet(self, df: pd.DataFrame, local_path: Path, s3_key: str | None = None) -> str:
        """Write ``df`` to local Parquet and (optionally) upload to S3."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False)

        if self._s3 is not None and s3_key:  # pragma: no cover - network
            try:
                self._s3.upload_file(str(local_path), self.cloud.aws_s3_bucket, s3_key)
                uri = f"s3://{self.cloud.aws_s3_bucket}/{s3_key}"
                logger.info("Uploaded %d rows -> %s", len(df), uri)
                return uri
            except Exception as exc:
                logger.warning("S3 upload failed, kept local copy (%s)", exc)

        logger.info("Wrote %d rows -> %s", len(df), local_path)
        return str(local_path)
