"""Regression coverage for cache-safe runner maintenance (no Docker daemon)."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerCachePolicyTest(unittest.TestCase):
    def test_cleanup_keeps_recent_cache_and_never_escalates_publish(self):
        for mode, free in (("--publish", 1), ("--publish", 50), ("--light", 50)):
            with self.subTest(mode=mode, free=free), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                log = path / "docker-calls"
                docker = path / "docker"
                docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\n')
                docker.chmod(0o755)
                df = path / "df"
                df.write_text(f'#!/bin/sh\nprintf "Filesystem 1K-blocks Used Available Use%% Mounted\\n/dev/test 100 50 50 {100-free}%% /\\n"\n')
                df.chmod(0o755)
                env = {**os.environ, "PATH": f"{path}:{os.environ['PATH']}",
                       "DOCKER_CALLS": str(log), "RUNNER_MIN_FREE_PCT": "15"}
                result = subprocess.run(
                    ["bash", str(ROOT / "scripts/github-runner-cleanup.sh"), mode],
                    env=env, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = log.read_text().splitlines()
                if mode == "--publish":
                    self.assertFalse(any(c.startswith(("builder prune", "image prune")) for c in calls))
                else:
                    self.assertIn("builder prune -f --filter until=168h", calls)
                    self.assertIn("image prune -f --filter until=168h", calls)

    def test_ci_and_scheduled_maintenance_use_cache_safe_modes(self):
        for name in ("ci.yml", "docker-publish.yml"):
            text = (ROOT / ".github/workflows" / name).read_text()
            self.assertIn("cleanup.sh --publish", text)
            self.assertNotIn("cleanup.sh --standard", text)
            self.assertNotIn("cleanup.sh --light", text)
        text = (ROOT / ".github/workflows/runner-maintenance.yml").read_text()
        self.assertIn("default: light", text)
        self.assertIn("inputs.mode || 'light'", text)
        publish = (ROOT / ".github/workflows/docker-publish.yml").read_text()
        self.assertNotIn('docker rmi "${primary}"', publish)


if __name__ == "__main__":
    unittest.main()

