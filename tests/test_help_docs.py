from __future__ import annotations

import html
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from survng.app.help_docs import (
    GUIDE_PAGES,
    _markdown_to_html,
    _rewrite_href,
    _safe_slug,
)
from survng.app.main import app


class HelpDocsTest(unittest.TestCase):
    def test_guide_home_and_pages_are_served(self) -> None:
        client = TestClient(app)
        home = client.get("/help")
        self.assertEqual(home.status_code, 200)
        self.assertIn("text/html", home.headers["content-type"])
        self.assertIn("Welcome to SurvNG", home.text)
        self.assertIn("Operator guide", home.text)

        for slug, label in GUIDE_PAGES:
            if slug == "index":
                continue
            with self.subTest(slug=slug):
                response = client.get(f"/help/{slug}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(html.escape(label), response.text)
                self.assertIn('class="help-article"', response.text)

    def test_reference_docs_are_served(self) -> None:
        client = TestClient(app)
        response = client.get("/help/reference/adaptive-motion")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Motion triggers and validation", response.text)
        self.assertIn("Technical reference", response.text)

    def test_help_stylesheet_is_served(self) -> None:
        client = TestClient(app)
        response = client.get("/help/assets/help.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.headers["content-type"])
        self.assertIn(".help-shell", response.text)
        home = client.get("/help")
        self.assertIn("/help/assets/help.css", home.text)

    def test_unknown_pages_return_404(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/help/not-a-real-page").status_code, 404)
        self.assertEqual(client.get("/help/reference/missing-doc").status_code, 404)
        with self.assertRaises(HTTPException) as raised:
            _safe_slug("../config")
        self.assertEqual(raised.exception.status_code, 404)

    def test_markdown_links_rewrite_to_help_routes(self) -> None:
        self.assertEqual(_rewrite_href("concepts.md", "/survng"), "/survng/help/concepts")
        self.assertEqual(_rewrite_href("index.md", "/survng"), "/survng/help")
        self.assertEqual(
            _rewrite_href("../adaptive-motion.md", "/survng"),
            "/survng/help/reference/adaptive-motion",
        )
        self.assertEqual(
            _rewrite_href("storage-retention.md", ""),
            "/help/reference/storage-retention",
        )
        self.assertEqual(_rewrite_href("/incidents", "/survng"), "/survng/incidents")
        self.assertTrue(_rewrite_href("https://example.com/x", "/survng").startswith("https://"))

    def test_markdown_renderer_handles_basics(self) -> None:
        html_body = _markdown_to_html(
            "# Title\n\nSee [Live](live.md) and `code`.\n\n- one\n- two\n\n"
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n```bash\necho hi\n```\n",
            "/survng",
        )
        self.assertIn('<h1 id="title">Title</h1>', html_body)
        self.assertIn('href="/survng/help/live"', html_body)
        self.assertIn("<code>code</code>", html_body)
        self.assertIn("<ul>", html_body)
        self.assertIn("<table>", html_body)
        self.assertIn("echo hi", html_body)


if __name__ == "__main__":
    unittest.main()
