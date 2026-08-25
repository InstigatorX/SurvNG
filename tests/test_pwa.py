from __future__ import annotations

import unittest

from survng import __version__
from survng.app.pwa import (
    normalize_service_worker_cache_version,
    service_worker_allowed_scope,
    service_worker_script,
    web_app_manifest,
)


class ProgressiveWebAppTest(unittest.TestCase):
    def test_package_version_is_v1(self) -> None:
        self.assertEqual(__version__, "1.0.0")

    def test_manifest_uses_configured_base_path(self) -> None:
        manifest = web_app_manifest("/survng")
        self.assertEqual(manifest["start_url"], "/survng/")
        self.assertEqual(manifest["scope"], "/survng/")
        self.assertEqual(manifest["display"], "standalone")
        self.assertTrue(any(icon["src"].startswith("/survng/static/") for icon in manifest["icons"]))

    def test_manifest_supports_root_installs(self) -> None:
        manifest = web_app_manifest("")
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(manifest["scope"], "/")
        self.assertTrue(any(icon["src"].startswith("/static/") for icon in manifest["icons"]))

    def test_service_worker_never_claims_api_routes(self) -> None:
        script = service_worker_script("/survng", cache_version="51894a9deadbeef")
        self.assertIn('path.includes("/api/")', script)
        self.assertIn("/survng/static/assets/", script)
        self.assertIn("survng-static-51894a9deadbeef", script)
        self.assertNotIn("survng-static-v1", script)
        self.assertEqual(service_worker_allowed_scope("/survng"), "/survng/")
        self.assertEqual(service_worker_allowed_scope(""), "/")

    def test_service_worker_cache_version_is_sanitized(self) -> None:
        self.assertEqual(normalize_service_worker_cache_version(""), "v2")
        self.assertEqual(normalize_service_worker_cache_version("ABC_def/01"), "abc-def-01")
        self.assertEqual(normalize_service_worker_cache_version("!!!"), "v2")
        self.assertIn(
            "survng-static-v2",
            service_worker_script("/survng"),
        )


if __name__ == "__main__":
    unittest.main()
