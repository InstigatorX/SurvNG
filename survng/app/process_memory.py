from __future__ import annotations

import ctypes
import gc
from pathlib import Path
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


def process_memory_status(
    *,
    status_path: Path = Path("/proc/self/status"),
    smaps_rollup_path: Path = Path("/proc/self/smaps_rollup"),
    fd_path: Path = Path("/proc/self/fd"),
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
        "malloc": _malloc_status(),
    }


def _status_count(path: Path, key: str) -> int:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return 0
