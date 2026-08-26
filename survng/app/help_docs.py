"""Online user documentation served at /help."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

GUIDE_DIR = Path("docs/guide")
REFERENCE_DIR = Path("docs")

GUIDE_PAGES = (
    ("index", "Welcome"),
    ("concepts", "Concepts"),
    ("getting-started", "Getting started"),
    ("live", "Live view"),
    ("incidents", "Incidents"),
    ("timeline", "Timeline & exports"),
    ("search", "Search"),
    ("people", "People"),
    ("admin", "Admin"),
    ("cameras", "Cameras"),
    ("motion-detection", "Motion & detection"),
    ("zones", "Zones"),
    ("storage", "Recordings & storage"),
    ("assistant", "AI assistant"),
    ("integrations", "Integrations"),
    ("access", "Access"),
    ("reverse-proxy", "Reverse proxy"),
    ("api", "HTTP API"),
)

_GUIDE_TITLES = dict(GUIDE_PAGES)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_UL_RE = re.compile(r"^[-*]\s+(.+)$")
_OL_RE = re.compile(r"^(\d+)\.\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_HR_RE = re.compile(r"^---+\s*$")
_ASSET_MEDIA_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}


@dataclass(frozen=True, slots=True)
class HelpRouteDependencies:
    base_path: Callable[[], str]


def create_help_router(deps: HelpRouteDependencies) -> APIRouter:
    router = APIRouter(tags=["help"])

    @router.get("/help", include_in_schema=False)
    @router.get("/help/", include_in_schema=False)
    def help_home() -> HTMLResponse:
        return _render_guide("index", deps.base_path())

    @router.get("/help/assets/help.css", include_in_schema=False)
    def help_stylesheet() -> Response:
        return _help_asset_response("help.css")

    @router.get("/help/assets/{asset_path:path}", include_in_schema=False)
    def help_asset(asset_path: str) -> Response:
        return _help_asset_response(asset_path)

    @router.get("/help/reference/{page}", include_in_schema=False)
    def help_reference_page(page: str) -> HTMLResponse:
        slug = _safe_slug(page)
        path = REFERENCE_DIR / f"{slug}.md"
        if (
            not path.is_file()
            or not _is_under(path, REFERENCE_DIR)
            or _is_under(path, GUIDE_DIR)
        ):
            raise HTTPException(status_code=404, detail="Reference page not found")
        source = path.read_text(encoding="utf-8")
        title = _title_from_markdown(source, slug)
        body = _markdown_to_html(source, deps.base_path())
        return HTMLResponse(
            _page_shell(
                title=title,
                body_html=body,
                base_path=deps.base_path(),
                active_slug=None,
                is_reference=True,
            )
        )

    @router.get("/help/{page}", include_in_schema=False)
    def help_guide_page(page: str) -> HTMLResponse:
        slug = _safe_slug(page)
        if slug not in _GUIDE_TITLES:
            raise HTTPException(status_code=404, detail="Help page not found")
        return _render_guide(slug, deps.base_path())

    return router


def _help_asset_response(asset_path: str) -> Response:
    normalized = str(asset_path or "").replace("\\", "/").lstrip("/")
    if (
        not normalized
        or normalized.startswith("../")
        or "/../" in f"/{normalized}/"
        or normalized.startswith("/")
    ):
        raise HTTPException(status_code=404, detail="Help asset not found")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", normalized):
        raise HTTPException(status_code=404, detail="Help asset not found")
    path = (GUIDE_DIR / normalized).resolve()
    if not path.is_file() or not _is_under(path, GUIDE_DIR):
        raise HTTPException(status_code=404, detail="Help asset not found")
    media_type = _ASSET_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=404, detail="Help asset not found")
    return Response(
        path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


def _safe_slug(value: str) -> str:
    slug = str(value or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", slug):
        raise HTTPException(status_code=404, detail="Help page not found")
    return slug


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _render_guide(slug: str, base_path: str) -> HTMLResponse:
    path = GUIDE_DIR / ("index.md" if slug == "index" else f"{slug}.md")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Help page not found")
    source = path.read_text(encoding="utf-8")
    title = _GUIDE_TITLES.get(slug) or _title_from_markdown(source, slug)
    body = _markdown_to_html(source, base_path)
    return HTMLResponse(
        _page_shell(
            title=title,
            body_html=body,
            base_path=base_path,
            active_slug=slug,
            is_reference=False,
        )
    )


def _title_from_markdown(source: str, fallback: str) -> str:
    for line in source.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("-", " ").title()


def _markdown_to_html(source: str, base_path: str) -> str:
    text = source.replace("\r\n", "\n")
    fences: list[str] = []

    def stash_fence(match: re.Match[str]) -> str:
        language = html.escape(match.group(1) or "")
        code = html.escape(match.group(2).rstrip("\n"))
        fences.append(
            f'<pre class="help-code"><code class="language-{language}">{code}</code></pre>'
        )
        return f"\x00FENCE{len(fences) - 1}\x00"

    text = _FENCE_RE.sub(stash_fence, text)
    blocks: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    list_items: list[str] = []
    table_rows: list[list[str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        blocks.append(f"<p>{_inline(' '.join(paragraph), base_path)}</p>")
        paragraph = []

    def flush_list() -> None:
        nonlocal list_kind, list_items
        if not list_kind or not list_items:
            list_kind = None
            list_items = []
            return
        tag = "ul" if list_kind == "ul" else "ol"
        items = "".join(f"<li>{item}</li>" for item in list_items)
        blocks.append(f"<{tag}>{items}</{tag}>")
        list_kind = None
        list_items = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        header, *rows = table_rows
        # Skip markdown separator row like |---|---|
        if rows and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in rows[0]):
            rows = rows[1:]
        thead = "".join(f"<th>{_inline(cell, base_path)}</th>" for cell in header)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{_inline(cell, base_path)}</td>" for cell in row)
            + "</tr>"
            for row in rows
        )
        blocks.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>")
        table_rows = []

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        fence_match = re.fullmatch(r"\x00FENCE(\d+)\x00", line.strip())
        if fence_match:
            flush_paragraph()
            flush_list()
            flush_table()
            blocks.append(fences[int(fence_match.group(1))])
            continue
        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_table()
            continue
        if _HR_RE.match(line):
            flush_paragraph()
            flush_list()
            flush_table()
            blocks.append("<hr />")
            continue
        image = _IMAGE_RE.match(line.strip())
        if image:
            flush_paragraph()
            flush_list()
            flush_table()
            alt = html.escape(image.group(1))
            src = html.escape(_rewrite_href(image.group(2).strip(), base_path), quote=True)
            caption = f"<figcaption>{alt}</figcaption>" if image.group(1).strip() else ""
            blocks.append(
                f'<figure class="help-figure"><img src="{src}" alt="{alt}" loading="lazy" />{caption}</figure>'
            )
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading.group(1))
            content = _inline(heading.group(2), base_path)
            anchor = _anchor_id(heading.group(2))
            blocks.append(f'<h{level} id="{anchor}">{content}</h{level}>')
            continue
        if _TABLE_ROW_RE.match(line):
            flush_paragraph()
            flush_list()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            table_rows.append(cells)
            continue
        ul = _UL_RE.match(line)
        if ul:
            flush_paragraph()
            flush_table()
            if list_kind not in (None, "ul"):
                flush_list()
            list_kind = "ul"
            list_items.append(_inline(ul.group(1), base_path))
            continue
        ol = _OL_RE.match(line)
        if ol:
            flush_paragraph()
            flush_table()
            if list_kind not in (None, "ol"):
                flush_list()
            list_kind = "ol"
            list_items.append(_inline(ol.group(2), base_path))
            continue
        flush_list()
        flush_table()
        paragraph.append(line.strip())

    flush_paragraph()
    flush_list()
    flush_table()
    return "\n".join(blocks)


def _inline(text: str, base_path: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _LINK_RE.finditer(text):
        pieces.append(html.escape(text[cursor:match.start()]))
        label = html.escape(match.group(1))
        href = html.escape(_rewrite_href(match.group(2).strip(), base_path), quote=True)
        pieces.append(f'<a href="{href}">{label}</a>')
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    result = "".join(pieces)
    result = _BOLD_RE.sub(r"<strong>\1</strong>", result)
    result = _INLINE_CODE_RE.sub(r"<code>\1</code>", result)
    return result


def _rewrite_href(target: str, base_path: str) -> str:
    if (
        target.startswith("#")
        or target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
    ):
        return target
    prefix = base_path.rstrip("/")
    help_root = f"{prefix}/help" if prefix else "/help"

    # Guide images: images/live-command-center.png
    if re.fullmatch(r"images/[A-Za-z0-9_-]+\.(?:png|jpe?g|webp|gif|svg)", target):
        return f"{help_root}/assets/{target}"

    # Guide or same-folder markdown: concepts.md, ./getting-started.md, storage.md
    if re.fullmatch(r"\.?/?(?:[A-Za-z0-9_-]+\.md)", target):
        name = Path(target).stem
        if name == "index":
            return help_root
        if name in _GUIDE_TITLES:
            return f"{help_root}/{name}"
        return f"{help_root}/reference/{name}"

    # Technical reference from the guide tree: ../storage.md
    if target.startswith("../") and target.endswith(".md") and "/" not in target[3:]:
        name = Path(target).stem
        return f"{help_root}/reference/{name}"

    # Already a help path
    if target.startswith("/help"):
        return f"{prefix}{target}" if prefix and not target.startswith(prefix) else target

    # App routes such as /incidents
    if target.startswith("/"):
        return f"{prefix}{target}" if prefix else target

    return target


def _anchor_id(heading: str) -> str:
    value = heading.strip().lower()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"\s+", "-", value)
    return value or "section"


def _page_shell(
    *,
    title: str,
    body_html: str,
    base_path: str,
    active_slug: str | None,
    is_reference: bool,
) -> str:
    prefix = base_path.rstrip("/")
    help_home = f"{prefix}/help" if prefix else "/help"
    css_href = f"{prefix}/help/assets/help.css" if prefix else "/help/assets/help.css"
    app_home = f"{prefix}/" if prefix else "/"
    nav_items = []
    for slug, label in GUIDE_PAGES:
        href = help_home if slug == "index" else f"{help_home}/{slug}"
        current = ' aria-current="page"' if slug == active_slug else ""
        css = ' class="active"' if slug == active_slug else ""
        nav_items.append(f'<a href="{html.escape(href)}"{css}{current}>{html.escape(label)}</a>')
    nav_html = "\n".join(nav_items)
    banner = (
        '<p class="help-reference-banner">Additional detail for this topic.</p>'
        if is_reference
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} · SurvNG Help</title>
  <link rel="stylesheet" href="{html.escape(css_href)}" />
  <link rel="icon" href="{html.escape(prefix + '/static/favicon.svg' if prefix else '/static/favicon.svg')}" />
</head>
<body class="help-body">
  <a class="help-skip" href="#help-main">Skip to content</a>
  <div class="help-shell">
    <aside class="help-sidebar" aria-label="Help topics">
      <a class="help-brand" href="{html.escape(help_home)}">
        <strong>SurvNG Help</strong>
        <span>Documentation</span>
      </a>
      <nav class="help-nav">{nav_html}</nav>
      <a class="help-back" href="{html.escape(app_home)}">← Back to SurvNG</a>
    </aside>
    <main id="help-main" class="help-main">
      {banner}
      <article class="help-article">
        {body_html}
      </article>
    </main>
  </div>
</body>
</html>
"""
