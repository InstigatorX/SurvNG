"""Static file serving for the built interface."""

from __future__ import annotations

import gzip
import threading
from pathlib import Path
from typing import Any

import anyio
from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

# The build names every file under assets/ after its content, so a name never
# refers to different bytes and can be kept by the browser without asking.
_IMMUTABLE = "public, max-age=31536000, immutable"
_TEXT_SUFFIXES = frozenset({".js", ".mjs", ".css", ".svg", ".json", ".webmanifest", ".map"})
_MINIMUM_BYTES = 1024
_MAXIMUM_ENTRIES = 256


def _accepts_gzip(value: str) -> bool:
    for item in value.split(","):
        coding, _, parameters = item.strip().partition(";")
        if coding.strip().lower() not in {"gzip", "*"}:
            continue
        quality = 1.0
        for parameter in parameters.split(";"):
            name, _, raw = parameter.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(raw.strip())
                except ValueError:
                    quality = 0.0
        if quality > 0.0:
            return True
    return False


class InterfaceStaticFiles(StaticFiles):
    """Serve build output compressed, and let hashed assets be cached for good.

    Scripts and stylesheets are compressed once per file version and kept in
    memory; the JSON compression middleware deliberately leaves them alone.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compressed: dict[str, tuple[tuple[int, int], bytes]] = {}
        self._guard = threading.Lock()

    def _gzip(self, path: str, version: tuple[int, int]) -> bytes:
        with self._guard:
            cached = self._compressed.get(path)
            if cached is not None and cached[0] == version:
                return cached[1]
        body = gzip.compress(Path(path).read_bytes(), compresslevel=6, mtime=0)
        with self._guard:
            if len(self._compressed) >= _MAXIMUM_ENTRIES:
                self._compressed.clear()
            self._compressed[path] = (version, body)
        return body

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code not in (200, 304):
            return response
        relative = path.replace("\\", "/").lstrip("/")
        if relative.startswith("assets/"):
            response.headers["Cache-Control"] = _IMMUTABLE
        if Path(relative).suffix.lower() not in _TEXT_SUFFIXES:
            return response
        response.headers["Vary"] = "Accept-Encoding"
        request = Headers(scope=scope)
        stat = getattr(response, "stat_result", None)
        if (
            response.status_code != 200
            or not isinstance(response, FileResponse)
            or stat is None
            or stat.st_size < _MINIMUM_BYTES
            or scope.get("method") != "GET"
            or "range" in request
            or not _accepts_gzip(request.get("accept-encoding", ""))
        ):
            return response
        body = await anyio.to_thread.run_sync(self._gzip, str(response.path), (stat.st_mtime_ns, stat.st_size))
        headers = {
            name: value for name, value in response.headers.items()
            if name.lower() not in {"content-length", "accept-ranges", "content-encoding"}
        }
        headers["Content-Encoding"] = "gzip"
        # The compressed form is a different representation of the same file.
        etag = headers.get("etag", "")
        if etag and not etag.startswith("W/"):
            headers["etag"] = f"W/{etag}"
        return Response(body, media_type=response.media_type, headers=headers)
