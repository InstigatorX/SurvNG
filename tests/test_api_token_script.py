from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ApiTokenScriptTest(unittest.TestCase):
    def run_cli(self, config_path: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
                "scripts/create-api-token.py",
                *arguments,
                "--config",
                str(config_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=check,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "."},
        )

    def test_cli_prints_secret_once_and_persists_only_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"cameras": []}), encoding="utf-8")
            result = self.run_cli(config_path,
                    "--id",
                    "home-assistant",
                    "--name",
                    "Home Assistant",
                    "--scope",
                    "read",
                    "--scope",
                    "camera:control",
                    "--enable",
            )
            token = result.stdout.strip().splitlines()[-1]
            saved_text = config_path.read_text(encoding="utf-8")
            saved = json.loads(saved_text)

        self.assertTrue(token.startswith("survng_"))
        self.assertNotIn(token, saved_text)
        self.assertTrue(saved["api_auth"]["enabled"])
        self.assertEqual(saved["api_auth"]["tokens"][0]["id"], "home-assistant")
        self.assertEqual(
            saved["api_auth"]["tokens"][0]["scopes"],
            ["read", "camera:control"],
        )
        self.assertEqual(len(saved["api_auth"]["tokens"][0]["token_hash"]), 64)

    def test_cli_lists_metadata_and_deletes_by_id_without_exposing_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"cameras": []}), encoding="utf-8")
            created = self.run_cli(config_path, "create", "--id", "ha", "--name", "Home Assistant", "--enable")
            raw_token = created.stdout.strip().splitlines()[-1]

            listed = self.run_cli(config_path, "list")
            self.assertIn("ha\tHome Assistant\tread", listed.stdout)
            self.assertNotIn(raw_token, listed.stdout)
            self.assertNotIn("token_hash", listed.stdout)

            deleted = self.run_cli(config_path, "delete", "--id", "ha")
            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertIn("Deleted API token: ha", deleted.stdout)
        self.assertEqual(saved["api_auth"]["tokens"], [])
        self.assertFalse(saved["api_auth"]["enabled"])


if __name__ == "__main__":
    unittest.main()
