"""Logging setup.

Human-readable by default; ``BWT_LOG_FORMAT=json`` switches to one JSON object
per line for log shippers. MNE is muted to WARNING because it is extremely
chatty at INFO and would drown out our own messages during a 105-subject load.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("BWT_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("BWT_LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # Third-party noise control.
    logging.getLogger("mne").setLevel(logging.WARNING)
    try:  # pragma: no cover - only present once mne is importable
        import mne

        mne.set_log_level("WARNING")
    except Exception:
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
