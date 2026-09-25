"""Cancellation scope for evidence work on the security worker.

The synchronous pipeline crosses adapters shared with initial detection. A
context-local token keeps cancellation attached to this attempt, without
mutating those adapters or putting process-local callbacks in durable payloads.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable
import subprocess
import time


class EvidenceWorkPreempted(Exception):
    """Cancel evidence work without treating it as a failed attempt."""


_stage_reporter: ContextVar[Callable[[str], None] | None] = ContextVar("evidence_stage_reporter", default=None)


_optional: ContextVar[bool] = ContextVar("evidence_optional", default=False)


_cancelled: ContextVar[Callable[[], bool] | None] = ContextVar("evidence_cancelled", default=None)


def report_evidence_stage(stage: str) -> None:
    callback = _stage_reporter.get()
    if callback is not None:
        callback(stage)


def evidence_work_active() -> bool:
    return _cancelled.get() is not None


def optional_evidence_work_active() -> bool:
    return _optional.get()


def evidence_cancelled() -> bool:
    callback = _cancelled.get()
    return bool(callback and callback())


def evidence_wait_timeout(timeout: float) -> float:
    """Poll cancellation when scoped; unscoped admission is unchanged."""
    return min(timeout, 0.05) if _cancelled.get() is not None else timeout


def check_evidence_cancellation() -> None:
    if evidence_cancelled():
        raise EvidenceWorkPreempted()


@contextmanager
def cancellable_evidence_work(
    cancelled: Callable[[], bool], *, optional: bool = True,
    stage_reporter: Callable[[str], None] | None = None,
):
    token = _cancelled.set(cancelled)
    optional_token = _optional.set(optional)
    stage_token = _stage_reporter.set(stage_reporter)
    try:
        check_evidence_cancellation()
        yield
        check_evidence_cancellation()
    finally:
        _cancelled.reset(token)
        _optional.reset(optional_token)
        _stage_reporter.reset(stage_token)


def run_evidence_process(command: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    """Drain or reap this attempt's decoder before releasing its capacity lease."""
    if not evidence_work_active():
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    check_evidence_cancellation()
    report_evidence_stage("frame_decode")
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                check_evidence_cancellation()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=evidence_wait_timeout(remaining))
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
