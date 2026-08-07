from __future__ import annotations

import unittest

from survng.app.main import (
    app,
    recording_exports_page,
    recording_search_page,
    recordings_page,
)


class FrontendRouteTest(unittest.TestCase):
    def test_recording_subpages_serve_the_recordings_application(self) -> None:
        for page in (recordings_page, recording_search_page, recording_exports_page):
            with self.subTest(page=page.__name__):
                response = page()
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.media_type)

    def test_smart_search_route_is_registered(self) -> None:
        paths = set(app.openapi()["paths"])

        self.assertIn("/recordings/search", paths)

    def test_frontend_routes_do_not_expose_template_paths_as_parameters(self) -> None:
        paths = app.openapi()["paths"]

        for path in ("/", "/recordings", "/recordings/search", "/config"):
            self.assertEqual(paths[path]["get"].get("parameters", []), [])


if __name__ == "__main__":
    unittest.main()
