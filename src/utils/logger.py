"""Console + file logging configured once for the whole platform."""

from __future__ import annotations

import logging
import sys

from src.config import LOG_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that writes to console and ``output/logs``."""
    global _CONFIGURED
    if not _CONFIGURED:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)

        file_handler = logging.FileHandler(LOG_DIR / "platform.log", encoding="utf-8")
        file_handler.setFormatter(formatter)

        root = logging.getLogger("claims")
        root.setLevel(logging.INFO)
        root.handlers.clear()
        root.addHandler(console)
        root.addHandler(file_handler)
        root.propagate = False
        _CONFIGURED = True

    return logging.getLogger(f"claims.{name}")
