"""Snowflake warehouse loader for fraud scores and compliance history.

Loads the scored claims and provider risk summaries into Snowflake (Dev Edition)
so downstream dbt models and the Power BI compliance dashboard can read curated
tables. When Snowflake credentials are absent, the curated tables are written to
local CSV/Parquet under ``output/scored`` so the dashboard layer still has data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG, SCORED_DIR
from src.utils.logger import get_logger

logger = get_logger("warehouse")


class WarehouseLoader:
    def __init__(self) -> None:
        self.cloud = CONFIG.cloud
        self._conn = None
        if self.cloud.snowflake_enabled:
            self._try_connect()

    def _try_connect(self) -> None:
        try:
            import snowflake.connector

            self._conn = snowflake.connector.connect(
                account=self.cloud.snowflake_account,
                user=self.cloud.snowflake_user,
                password=self.cloud.snowflake_password,
                database=self.cloud.snowflake_database,
                schema=self.cloud.snowflake_schema,
                warehouse=self.cloud.snowflake_warehouse,
            )
            logger.info("Snowflake connection established (db=%s)", self.cloud.snowflake_database)
        except Exception as exc:
            logger.warning("Snowflake unavailable, writing curated tables locally (%s)", exc)
            self._conn = None

    def load_table(self, df: pd.DataFrame, table_name: str) -> str:
        """Load a DataFrame into a Snowflake table (or local fallback)."""
        if self._conn is not None:  # pragma: no cover - network
            try:
                from snowflake.connector.pandas_tools import write_pandas

                write_pandas(self._conn, df, table_name.upper(), auto_create_table=True,
                             overwrite=True)
                logger.info("Loaded %d rows into Snowflake.%s", len(df), table_name)
                return f"snowflake://{self.cloud.snowflake_database}/{table_name}"
            except Exception as exc:
                logger.warning("Snowflake load failed, writing locally (%s)", exc)

        local_path = Path(SCORED_DIR) / f"{table_name}.parquet"
        df.to_parquet(local_path, index=False)
        df.to_csv(local_path.with_suffix(".csv"), index=False)
        logger.info("Wrote curated table %s -> %s", table_name, local_path)
        return str(local_path)

    def close(self) -> None:
        if self._conn is not None:  # pragma: no cover
            self._conn.close()
