from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.process_memory import process_memory_status


class ProcessMemoryStatusTest(unittest.TestCase):
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
