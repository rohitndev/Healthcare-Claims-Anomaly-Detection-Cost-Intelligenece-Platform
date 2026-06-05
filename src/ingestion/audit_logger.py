"""HIPAA PHI access audit sink.

Every access to a PHI identifier is recorded with a timestamp, the actor, the
action, and the affected record count, satisfying the HIPAA requirement to log
all PHI accesses (45 CFR 164.312(b)). Records are appended to a JSONL audit log
that the compliance reporter can summarise monthly.
"""

from __future__ import annotations

import getpass
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from src.config import AUDIT_DIR
from src.utils.logger import get_logger

logger = get_logger("audit")


class HIPAAAuditLogger:
    """Append-only audit log for PHI access events."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or (AUDIT_DIR / "phi_access_audit.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._actor = getpass.getuser()
        except Exception:  # pragma: no cover
            self._actor = "unknown"
        self._host = socket.gethostname()

    def log_access(
        self,
        action: str,
        phi_identifiers: list[str],
        record_count: int,
        purpose: str = "claims anomaly processing",
    ) -> dict:
        """Record a single PHI access event and return it."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": self._actor,
            "host": self._host,
            "action": action,
            "phi_identifiers": phi_identifiers,
            "record_count": int(record_count),
            "purpose": purpose,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        logger.info(
            "PHI %s | %d records | %d identifiers", action, record_count, len(phi_identifiers)
        )
        return event

    def summary(self) -> dict:
        """Aggregate the audit log into a compliance summary."""
        if not self.log_path.exists():
            return {"events": 0, "records_touched": 0}
        events = [json.loads(line) for line in self.log_path.read_text("utf-8").splitlines() if line]
        return {
            "events": len(events),
            "records_touched": sum(e["record_count"] for e in events),
            "actions": sorted({e["action"] for e in events}),
            "first_event": events[0]["timestamp"] if events else None,
            "last_event": events[-1]["timestamp"] if events else None,
        }
