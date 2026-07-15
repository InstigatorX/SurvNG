from __future__ import annotations

import unittest

from pydantic import ValidationError

from survng.app.config import AppConfig


class AppConfigTest(unittest.TestCase):
    def test_base_path_defaults_to_survng(self) -> None:
        self.assertEqual(AppConfig().base_path, "/survng")

    def test_base_path_is_normalized(self) -> None:
        self.assertEqual(AppConfig(base_path=" cameras/ ").base_path, "/cameras")
        self.assertEqual(AppConfig(base_path="/").base_path, "")
        self.assertEqual(AppConfig(base_path="").base_path, "")

    def test_base_path_rejects_query_or_fragment(self) -> None:
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/survng?mode=remote")
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/survng#remote")


if __name__ == "__main__":
    unittest.main()
