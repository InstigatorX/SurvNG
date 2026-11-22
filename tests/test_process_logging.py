"""Warnings and errors reach stderr (the journal) with redacted tracebacks."""

from __future__ import annotations

import io
import logging

from survng.app.process_logging import STDERR_HANDLER_NAME, install_stderr_log_handler


def test_errors_reach_stderr_with_redacted_traceback() -> None:
    root = logging.getLogger()
    # Another test may have run the service entry point, which installs it.
    previous = [h for h in root.handlers if getattr(h, "name", "") == STDERR_HANDLER_NAME]
    for existing in previous:
        root.removeHandler(existing)
    stream = io.StringIO()
    handler = install_stderr_log_handler(stream)
    try:
        assert install_stderr_log_handler(io.StringIO()) is handler
        logger = logging.getLogger("survng.app.test_process_logging")
        logger.info("routine detail")
        try:
            raise RuntimeError("open rtsp://admin:hunter2@camera.invalid/stream failed")
        except RuntimeError:
            logger.exception("Recording retention cycle failed")
    finally:
        root.removeHandler(handler)
        for existing in previous:
            root.addHandler(existing)

    output = stream.getvalue()
    assert "routine detail" not in output
    assert "ERROR survng.app.test_process_logging: Recording retention cycle failed" in output
    assert "Traceback (most recent call last)" in output
    assert "RuntimeError" in output
    assert "hunter2" not in output
