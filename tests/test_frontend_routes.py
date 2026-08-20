from __future__ import annotations

import unittest

from survng.app.main import (
    app,
    manager,
    recording_exports_page,
    recording_search_page,
    recordings_page,
    search_page,
    timeline_exports_page,
    timeline_page,
    exports_page,
)


class FrontendRouteTest(unittest.TestCase):
    def test_imported_application_uses_isolated_test_runtime(self) -> None:
        self.assertIn("survng-pytest-", str(manager.database_dir))

    def test_recording_subpages_serve_the_recordings_application(self) -> None:
        for page in (recordings_page, recording_search_page, recording_exports_page):
            with self.subTest(page=page.__name__):
                response = page()
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.media_type)

    def test_smart_search_route_is_registered(self) -> None:
        paths = set(app.openapi()["paths"])

        self.assertIn("/recordings/search", paths)
        self.assertIn("/search", paths)
        self.assertIn("/timeline", paths)
        self.assertIn("/timeline/exports", paths)
        self.assertIn("/exports", paths)
        self.assertIn("/people", paths)
        self.assertIn("/admin", paths)

    def test_canonical_timeline_routes_serve_the_recordings_application(self) -> None:
        for page in (timeline_page, timeline_exports_page, exports_page, search_page):
            with self.subTest(page=page.__name__):
                response = page()
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.media_type)

    def test_frontend_routes_do_not_expose_template_paths_as_parameters(self) -> None:
        paths = app.openapi()["paths"]

        for path in ("/", "/recordings", "/recordings/search", "/config", "/timeline", "/exports", "/search", "/admin", "/people"):
            self.assertEqual(paths[path]["get"].get("parameters", []), [])


if __name__ == "__main__":
    unittest.main()
