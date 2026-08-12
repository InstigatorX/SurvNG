from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ApiTokenScriptTest(unittest.TestCase):
    def test_cli_prints_secret_once_and_persists_only_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"cameras": []}), encoding="utf-8")
            result = subprocess.run(
                [
                    str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
                    "scripts/create-api-token.py",
                    "--config",
                    str(config_path),
                    "--id",
                    "home-assistant",
                    "--name",
                    "Home Assistant",
                    "--scope",
                    "read",
                    "--scope",
                    "camera:control",
                    "--enable",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": "."},
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


if __name__ == "__main__":
    unittest.main()
