from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.process_memory import (
    AllocatorMemoryTrimmer,
    AllocatorTrimPolicy,
    process_memory_status,
)


class ProcessMemoryStatusTest(unittest.TestCase):
    def test_allocator_trimmer_requires_quiet_threshold_and_interval(self) -> None:
        calls: list[str] = []
        trimmer = AllocatorMemoryTrimmer(
            AllocatorTrimPolicy(
                minimum_free_bytes=500,
                minimum_interval_seconds=300.0,
                quiet_seconds=30.0,
            ),
            trim=lambda: calls.append("trim") is None,
            monotonic=lambda: 0.0,
        )
        memory = {"rss_bytes": 1_000, "malloc": {"free_bytes": 600}}

        trimmer.observe_idle(True, now=100.0)
        not_quiet = trimmer.maybe_trim(memory, now=120.0)
        trimmed = trimmer.maybe_trim(
            memory,
            now=130.0,
            memory_after=lambda: {"rss_bytes": 700, "malloc": {}},
        )
        rate_limited = trimmer.maybe_trim(memory, now=200.0)

        self.assertEqual(not_quiet["last_skip_reason"], "busy_or_not_quiet")
        self.assertEqual(calls, ["trim"])
        self.assertEqual(trimmed["attempts"], 1)
        self.assertEqual(trimmed["reclaimed_total_bytes"], 300)
        self.assertEqual(rate_limited["last_skip_reason"], "rate_limited")

    def test_allocator_trimmer_resets_quiet_window_when_busy(self) -> None:
        trimmer = AllocatorMemoryTrimmer(
            AllocatorTrimPolicy(minimum_free_bytes=1, quiet_seconds=30.0),
            trim=lambda: True,
            monotonic=lambda: 0.0,
        )
        memory = {"rss_bytes": 100, "malloc": {"free_bytes": 50}}

        trimmer.observe_idle(True, now=100.0)
        trimmer.observe_idle(False, now=120.0)
        trimmer.observe_idle(True, now=125.0)
        status = trimmer.maybe_trim(memory, now=150.0)

        self.assertEqual(status["attempts"], 0)
        self.assertEqual(status["last_skip_reason"], "busy_or_not_quiet")

    def test_reads_proc_status_smaps_and_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status = root / "status"
            smaps = root / "smaps_rollup"
            descriptors = root / "fd"
            descriptors.mkdir()
            (descriptors / "1").touch()
            (descriptors / "2").touch()
            status.write_text(
                "VmRSS:\t100 kB\nVmHWM:\t120 kB\nRssAnon:\t80 kB\n"
                "RssFile:\t15 kB\nRssShmem:\t5 kB\nVmSize:\t500 kB\n"
                "VmData:\t300 kB\nThreads:\t9\n",
                encoding="utf-8",
            )
            smaps.write_text(
                "Pss:\t90 kB\nPss_Anon:\t75 kB\nPrivate_Dirty:\t70 kB\n"
                "AnonHugePages:\t20 kB\n",
                encoding="utf-8",
            )

            with patch(
                "survng.app.process_memory._malloc_status",
                return_value={
                    "arena_bytes": 10,
                    "allocated_bytes": 8,
                    "free_bytes": 2,
                    "mmap_bytes": 4,
                    "mmap_regions": 1,
                },
            ):
                result = process_memory_status(
                    status_path=status,
                    smaps_rollup_path=smaps,
                    fd_path=descriptors,
                )

            self.assertEqual(result["rss_bytes"], 100 * 1024)
            self.assertEqual(result["anonymous_pss_bytes"], 75 * 1024)
            self.assertEqual(result["threads"], 9)
            self.assertEqual(result["file_descriptors"], 2)
            self.assertEqual(result["malloc"]["allocated_bytes"], 8)
