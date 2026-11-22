"""Process-level log output for the SurvNG service.

The in-memory handler in ``main`` feeds the logs page but keeps only an
exception's last line. Warnings and errors also go to stderr, which systemd
records in the journal with full tracebacks, so failures can be diagnosed
after the in-memory buffer has rolled over.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from .security import redact_secret_text

STDERR_HANDLER_NAME = "survng-stderr-log"


class RedactingFormatter(logging.Formatter):
    """Redacts credentials from the whole record, traceback included."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secret_text(super().format(record))


def install_stderr_log_handler(stream: TextIO | None = None) -> logging.Handler:
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "name", "") == STDERR_HANDLER_NAME:
            return handler
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.name = STDERR_HANDLER_NAME
    handler.setLevel(logging.WARNING)
    # The journal timestamps each line itself.
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return handler
