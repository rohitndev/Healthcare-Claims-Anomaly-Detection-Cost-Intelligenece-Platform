"""Optional PySpark session factory with a transparent pandas fallback.

The platform is designed to run on PySpark (Databricks Community Edition in
production). To guarantee the pipeline also runs on a bare laptop without a
configured Spark/Hadoop environment, :func:`get_spark` returns ``None`` whenever
Spark cannot be initialised, and every processing stage provides an equivalent
pandas implementation.
"""

from __future__ import annotations

from typing import Optional

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger("spark")

_SESSION = None
_ATTEMPTED = False


def get_spark(app_name: str = "claims-intelligence") -> Optional["object"]:
    """Return a SparkSession when available and enabled, otherwise ``None``."""
    global _SESSION, _ATTEMPTED

    if not CONFIG.use_spark:
        return None

    if _ATTEMPTED:
        return _SESSION

    _ATTEMPTED = True
    try:
        from pyspark.sql import SparkSession

        _SESSION = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
        _SESSION.sparkContext.setLogLevel("ERROR")
        logger.info("Spark session started (version %s)", _SESSION.version)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Spark unavailable, using pandas engine instead (%s)", exc)
        _SESSION = None

    return _SESSION


def engine_name() -> str:
    """Human-readable name of the active execution engine."""
    return "PySpark" if get_spark() is not None else "pandas"
