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
        paths = {
            path
            for route in app.routes
            if (path := getattr(route, "path", None)) is not None
        }

        self.assertIn("/recordings/search", paths)


if __name__ == "__main__":
    unittest.main()
