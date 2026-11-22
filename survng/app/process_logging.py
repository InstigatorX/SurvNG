"""Process-level log output for the SurvNG service.

The in-memory handler in ``main`` feeds the logs page but keeps only an
exception's last line. Warnings and errors also go to stderr, which systemd
records in the journal with full tracebacks, so failures can be diagnosed
after the in-memory buffer has rolled over.

The stderr handler is part of the logging configuration handed to uvicorn,
so uvicorn's own ``dictConfig`` creates it, not something installed earlier
that the configuration then closes. Every formatter redacts credentials.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter, DefaultFormatter

from .security import redact_secret_text

STDERR_HANDLER_NAME = "survng_stderr"
# Per call site: this many records per window reach stderr, the rest are
# counted and reported once the window ends. FFmpeg stderr relays and retry
# loops can log continuously, and journald drops a service's messages once
# its rate limit is hit, including the tracebacks this output exists for.
RATE_LIMIT_BURST = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_SITES = 1024


class RedactingFormatter(logging.Formatter):
    """Redacts credentials from the whole record, traceback included."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secret_text(super().format(record))


class RedactingDefaultFormatter(DefaultFormatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secret_text(super().format(record))


class RedactingAccessFormatter(AccessFormatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secret_text(super().format(record))


class SiteRateLimitFilter(logging.Filter):
    """Pass at most ``burst`` records per call site per window.

    When a window that suppressed records ends, the next record from that
    site carries the count, so nothing is silently lost.
    """

    def __init__(
        self,
        burst: int = RATE_LIMIT_BURST,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        max_sites: int = RATE_LIMIT_MAX_SITES,
    ) -> None:
        super().__init__()
        self.burst = max(1, int(burst))
        self.window_seconds = float(window_seconds)
        self.max_sites = max(1, int(max_sites))
        self._lock = threading.Lock()
        # site -> [window start, records passed, records suppressed]
        self._sites: dict[tuple[str, int], list[float]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        site = (record.pathname, record.lineno)
        now = time.monotonic()
        with self._lock:
            state = self._sites.get(site)
            if state is None or now - state[0] >= self.window_seconds:
                suppressed = int(state[2]) if state is not None else 0
                if state is None and len(self._sites) >= self.max_sites:
                    oldest = min(self._sites, key=lambda key: self._sites[key][0])
                    del self._sites[oldest]
                self._sites[site] = [now, 1, 0]
            elif state[1] < self.burst:
                state[1] += 1
                suppressed = 0
            else:
                state[2] += 1
                return False
        if suppressed:
            record.msg = (
                f"{record.getMessage()} [{suppressed} similar messages from this "
                f"site suppressed in the previous {self.window_seconds:.0f} s]"
            )
            record.args = None
        return True


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's default logging configuration plus SurvNG's stderr output."""
    config = copy.deepcopy(LOGGING_CONFIG)
    config["formatters"]["default"]["()"] = f"{__name__}.RedactingDefaultFormatter"
    config["formatters"]["access"]["()"] = f"{__name__}.RedactingAccessFormatter"
    config["formatters"]["survng"] = {
        "()": f"{__name__}.RedactingFormatter",
        # The journal timestamps each line itself.
        "fmt": "%(levelname)s %(name)s: %(message)s",
    }
    config.setdefault("filters", {})["survng_rate_limit"] = {
        "()": f"{__name__}.SiteRateLimitFilter",
    }
    config["handlers"][STDERR_HANDLER_NAME] = {
        "class": "logging.StreamHandler",
        "formatter": "survng",
        "filters": ["survng_rate_limit"],
        "stream": "ext://sys.stderr",
        "level": "WARNING",
    }
    config["root"] = {"handlers": [STDERR_HANDLER_NAME], "level": "INFO"}
    return config
