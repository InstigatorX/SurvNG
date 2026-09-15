"""Cancellation scope for optional evidence work on the security worker.

The synchronous pipeline crosses adapters shared with initial detection. A
context-local token keeps cancellation attached to this attempt, without
mutating those adapters or putting process-local callbacks in durable payloads.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable


class EvidenceWorkPreempted(Exception):
    """Yield optional work; this is not a failed evidence attempt."""


_cancelled: ContextVar[Callable[[], bool] | None] = ContextVar("evidence_cancelled", default=None)


def evidence_work_active() -> bool:
    return _cancelled.get() is not None


def evidence_cancelled() -> bool:
    callback = _cancelled.get()
    return bool(callback and callback())


def evidence_wait_timeout(timeout: float) -> float:
    """Poll cancellation only for optional work; ordinary admission is unchanged."""
    return min(timeout, 0.05) if _cancelled.get() is not None else timeout


def check_evidence_cancellation() -> None:
    if evidence_cancelled():
        raise EvidenceWorkPreempted()


@contextmanager
def cancellable_evidence_work(cancelled: Callable[[], bool]):
    token = _cancelled.set(cancelled)
    try:
        check_evidence_cancellation()
        yield
        check_evidence_cancellation()
    finally:
        _cancelled.reset(token)
