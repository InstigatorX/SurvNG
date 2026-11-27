"""Compression and caching of the built interface's static files."""
import asyncio
import gzip

from survng.app.static_assets import InterfaceStaticFiles, _accepts_gzip


def serve(files, path, *, encoding="gzip, deflate, br", method="GET", extra=()):
    headers = [(b"accept-encoding", encoding.encode())] if encoding is not None else []
    headers += [(name.encode(), value.encode()) for name, value in extra]
    scope = {"type": "http", "method": method, "path": f"/{path}", "headers": headers}
    return asyncio.run(files.get_response(path, scope))


def site(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "App-abc123.js").write_text("export const value = 1;\n" * 400)
    (tmp_path / "assets" / "App-abc123.css").write_text(".a{color:red}\n" * 400)
    (tmp_path / "assets" / "tiny-abc123.js").write_text("1;")
    (tmp_path / "assets" / "font-abc123.woff2").write_bytes(bytes(range(256)) * 16)
    (tmp_path / "favicon.svg").write_text("<svg>" + "<g/>" * 600 + "</svg>")
    (tmp_path / "icon.png").write_bytes(bytes(range(256)) * 16)
    return InterfaceStaticFiles(directory=tmp_path)


def test_hashed_scripts_are_compressed_and_kept_by_the_browser(tmp_path):
    files = site(tmp_path)
    for name in ("assets/App-abc123.js", "assets/App-abc123.css"):
        response = serve(files, name)
        original = (tmp_path / name).read_bytes()
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["vary"] == "Accept-Encoding"
        assert gzip.decompress(response.body) == original
        assert len(response.body) < len(original) / 4
        assert int(response.headers["content-length"]) == len(response.body)
        assert response.headers["etag"].startswith('W/"')
    assert serve(files, "assets/App-abc123.js").media_type in {"text/javascript", "application/javascript"}


def test_compressed_bytes_are_built_once_per_file_version(tmp_path):
    files = site(tmp_path)
    first = serve(files, "assets/App-abc123.js").body
    assert serve(files, "assets/App-abc123.js").body is first
    script = tmp_path / "assets" / "App-abc123.js"
    script.write_text("export const value = 2;\n" * 500)
    changed = serve(files, "assets/App-abc123.js")
    assert gzip.decompress(changed.body) == script.read_bytes()


def test_clients_and_requests_that_cannot_take_gzip_get_the_file(tmp_path):
    files = site(tmp_path)
    for kwargs in ({"encoding": None}, {"encoding": "identity"}, {"encoding": "gzip;q=0"}, {"method": "HEAD"}, {"extra": [("range", "bytes=0-9")]}):
        response = serve(files, "assets/App-abc123.js", **kwargs)
        assert "content-encoding" not in response.headers, kwargs
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["vary"] == "Accept-Encoding"


def test_small_and_binary_files_are_not_compressed(tmp_path):
    files = site(tmp_path)
    assert "content-encoding" not in serve(files, "assets/tiny-abc123.js").headers
    font = serve(files, "assets/font-abc123.woff2")
    assert "content-encoding" not in font.headers and "vary" not in font.headers
    assert font.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_unhashed_files_are_compressed_but_still_revalidated(tmp_path):
    files = site(tmp_path)
    icon = serve(files, "favicon.svg")
    assert icon.headers["content-encoding"] == "gzip"
    assert "cache-control" not in icon.headers
    assert "cache-control" not in serve(files, "icon.png").headers
    # The validator of the compressed form still answers a conditional request.
    again = serve(files, "favicon.svg", extra=[("if-none-match", icon.headers["etag"])])
    assert again.status_code == 304


def test_accept_encoding_is_read_with_its_qualities():
    assert _accepts_gzip("gzip") and _accepts_gzip("br, gzip;q=0.5") and _accepts_gzip("*")
    assert not _accepts_gzip("") and not _accepts_gzip("br") and not _accepts_gzip("gzip;q=0") and not _accepts_gzip("gzip;q=bad")
