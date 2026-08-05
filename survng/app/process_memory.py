from __future__ import annotations

import ctypes
import gc
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from typing import Final


KIB: Final = 1024


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def _proc_kib(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * KIB
        except ValueError:
            continue
    return values


def _malloc_status() -> dict[str, int]:
    try:
        mallinfo2 = ctypes.CDLL(None).mallinfo2
        mallinfo2.argtypes = []
        mallinfo2.restype = _Mallinfo2
        status = mallinfo2()
    except (AttributeError, OSError):
        return {
            "arena_bytes": 0,
            "allocated_bytes": 0,
            "free_bytes": 0,
            "mmap_bytes": 0,
            "mmap_regions": 0,
        }
    return {
        "arena_bytes": int(status.arena),
        "allocated_bytes": int(status.uordblks),
        "free_bytes": int(status.fordblks),
        "mmap_bytes": int(status.hblkhd),
        "mmap_regions": int(status.hblks),
    }


def _malloc_trim() -> bool:
    """Ask glibc to return unused allocator pages to the kernel."""
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False


def process_memory_status_for_pid(pid: int) -> dict[str, int | dict[str, int]]:
    """Read low-overhead process diagnostics without querying its allocator."""
    root = Path("/proc") / str(max(0, int(pid)))
    result = process_memory_status(
        status_path=root / "status",
        smaps_rollup_path=root / "smaps_rollup",
        fd_path=root / "fd",
        include_malloc=False,
    )
    return result


@dataclass(frozen=True)
class AllocatorTrimPolicy:
    minimum_free_bytes: int = 512 * 1024 * 1024
    minimum_interval_seconds: float = 300.0
    quiet_seconds: float = 30.0


class AllocatorMemoryTrimmer:
    """Conservatively reclaim native arenas after high-resolution work is idle."""

    def __init__(
        self,
        policy: AllocatorTrimPolicy | None = None,
        *,
        trim: Callable[[], bool] = _malloc_trim,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or AllocatorTrimPolicy()
        self._trim = trim
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._idle_since: float | None = None
        self._last_attempt_at = 0.0
        self._status: dict[str, int | float | str | bool] = {
            "enabled": True,
            "minimum_free_bytes": self.policy.minimum_free_bytes,
            "minimum_interval_seconds": self.policy.minimum_interval_seconds,
            "quiet_seconds": self.policy.quiet_seconds,
            "attempts": 0,
            "successful_trims": 0,
            "reclaimed_total_bytes": 0,
            "last_reclaimed_bytes": 0,
            "last_trim_at": "",
            "last_skip_reason": "startup",
        }

    def observe_idle(self, idle: bool, *, now: float | None = None) -> None:
        current = self._monotonic() if now is None else now
        with self._lock:
            if idle:
                if self._idle_since is None:
                    self._idle_since = current
            else:
                self._idle_since = None
                self._status["last_skip_reason"] = "busy"

    def maybe_trim(
        self,
        memory: dict[str, int | dict[str, int]],
        *,
        now: float | None = None,
        memory_after: Callable[[], dict[str, int | dict[str, int]]] | None = None,
    ) -> dict[str, int | float | str | bool]:
        current = self._monotonic() if now is None else now
        malloc = memory.get("malloc")
        retained = int(malloc.get("free_bytes") or 0) if isinstance(malloc, dict) else 0
        with self._lock:
            idle_for = current - self._idle_since if self._idle_since is not None else 0.0
            if self._idle_since is None or idle_for < self.policy.quiet_seconds:
                self._status["last_skip_reason"] = "busy_or_not_quiet"
                return dict(self._status)
            if retained < self.policy.minimum_free_bytes:
                self._status["last_skip_reason"] = "below_threshold"
                return dict(self._status)
            if (
                int(self._status["attempts"]) > 0
                and current - self._last_attempt_at < self.policy.minimum_interval_seconds
            ):
                self._status["last_skip_reason"] = "rate_limited"
                return dict(self._status)
            self._last_attempt_at = current
            self._status["attempts"] = int(self._status["attempts"]) + 1

        before_rss = int(memory.get("rss_bytes") or 0)
        released = self._trim()
        after = (memory_after or process_memory_status)()
        reclaimed = max(0, before_rss - int(after.get("rss_bytes") or 0))
        with self._lock:
            if released or reclaimed:
                self._status["successful_trims"] = int(self._status["successful_trims"]) + 1
            self._status["reclaimed_total_bytes"] = (
                int(self._status["reclaimed_total_bytes"]) + reclaimed
            )
            self._status["last_reclaimed_bytes"] = reclaimed
            self._status["last_trim_at"] = datetime.now(timezone.utc).isoformat()
            self._status["last_skip_reason"] = ""
            return dict(self._status)

    def status(self) -> dict[str, int | float | str | bool]:
        with self._lock:
            return dict(self._status)


def process_memory_status(
    *,
    status_path: Path = Path("/proc/self/status"),
    smaps_rollup_path: Path = Path("/proc/self/smaps_rollup"),
    fd_path: Path = Path("/proc/self/fd"),
    include_malloc: bool = True,
) -> dict[str, int | dict[str, int]]:
    """Return low-overhead diagnostics that separate live and retained memory."""
    status = _proc_kib(status_path)
    smaps = _proc_kib(smaps_rollup_path)
    try:
        file_descriptors = sum(1 for _entry in fd_path.iterdir())
    except OSError:
        file_descriptors = 0
    gc_counts = gc.get_count()
    return {
        "rss_bytes": status.get("VmRSS", 0),
        "peak_rss_bytes": status.get("VmHWM", 0),
        "virtual_bytes": status.get("VmSize", 0),
        "data_bytes": status.get("VmData", 0),
        "anonymous_rss_bytes": status.get("RssAnon", 0),
        "file_rss_bytes": status.get("RssFile", 0),
        "shared_rss_bytes": status.get("RssShmem", 0),
        "pss_bytes": smaps.get("Pss", 0),
        "anonymous_pss_bytes": smaps.get("Pss_Anon", smaps.get("Anonymous", 0)),
        "private_dirty_bytes": smaps.get("Private_Dirty", 0),
        "anonymous_huge_pages_bytes": smaps.get("AnonHugePages", 0),
        "threads": _status_count(status_path, "Threads"),
        "file_descriptors": file_descriptors,
        "gc_generation_0": int(gc_counts[0]),
        "gc_generation_1": int(gc_counts[1]),
        "gc_generation_2": int(gc_counts[2]),
        "malloc": _malloc_status() if include_malloc else {},
    }


def _status_count(path: Path, key: str) -> int:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return 0
