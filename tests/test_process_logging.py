"""Warnings and errors reach stderr (the journal), redacted and rate-limited."""

from __future__ import annotations

import io
import logging
import logging.config

import pytest

from survng.app.process_logging import (
    STDERR_HANDLER_NAME,
    SiteRateLimitFilter,
    uvicorn_log_config,
)


@pytest.fixture
def configured_stderr():
    """Apply the service's logging config, then restore logging as it was."""
    names = ("", "uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {
        name: (logging.getLogger(name).handlers[:], logging.getLogger(name).level,
               logging.getLogger(name).propagate)
        for name in names
    }
    logging.config.dictConfig(uvicorn_log_config())
    root = logging.getLogger()
    handler = next(h for h in root.handlers if h.name == STDERR_HANDLER_NAME)
    stream = io.StringIO()
    handler.setStream(stream)
    try:
        yield stream
    finally:
        for name, (handlers, level, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate


def test_errors_reach_stderr_with_redacted_traceback(configured_stderr) -> None:
    logger = logging.getLogger("survng.app.test_process_logging")
    logger.info("routine detail")
    try:
        raise RuntimeError("open rtsp://admin:hunter2@camera.invalid/stream failed")
    except RuntimeError:
        logger.exception("Recording retention cycle failed")

    output = configured_stderr.getvalue()
    assert "routine detail" not in output
    assert "ERROR survng.app.test_process_logging: Recording retention cycle failed" in output
    assert "Traceback (most recent call last)" in output
    assert "RuntimeError" in output
    assert "hunter2" not in output


def test_uvicorn_formatters_redact(configured_stderr) -> None:
    formatter = logging.getLogger("uvicorn").handlers[0].formatter
    record = logging.LogRecord(
        "uvicorn.error", logging.ERROR, __file__, 1,
        "relay to rtsp://admin:hunter2@camera.invalid failed", None, None,
    )

    assert "hunter2" not in formatter.format(record)


def test_a_repeating_site_is_rate_limited_and_the_count_reported() -> None:
    limiter = SiteRateLimitFilter(burst=3, window_seconds=60.0)

    def record(lineno: int) -> logging.LogRecord:
        return logging.LogRecord("x", logging.WARNING, "/x.py", lineno, "ffmpeg %s", ("line",), None)

    passed = [limiter.filter(record(10)) for _ in range(10)]
    other_site = limiter.filter(record(20))
    assert passed == [True] * 3 + [False] * 7
    assert other_site

    # The next window reports what the last one dropped.
    limiter._sites[("/x.py", 10)][0] -= 61.0
    following = record(10)
    assert limiter.filter(following)
    assert following.getMessage() == (
        "ffmpeg line [7 similar messages from this site suppressed in the previous 60 s]"
    )
