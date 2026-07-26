from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from .config import CameraConfig


class Go2RtcError(RuntimeError):
    pass


@dataclass(frozen=True)
class Go2RtcStream:
    host: str
    name: str


class Go2RtcAdapter:
    def __init__(self, api_port: int = 1984, timeout: float = 5.0, cache_seconds: float = 5.0) -> None:
        self.api_port = int(api_port)
        self.timeout = max(1.0, float(timeout))
        self.cache_seconds = max(1.0, float(cache_seconds))
        self._streams_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._stream_locks: dict[str, threading.Lock] = {}
        self._compatibility_locks: dict[tuple[str, str], threading.Lock] = {}
        self._lock = threading.RLock()

    def stream(self, camera: CameraConfig, source: str = "live") -> Go2RtcStream:
        parsed = urlsplit(camera.source_url(camera.normalized_source(source)))
        stream_name = parsed.path.strip("/")
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname or not stream_name:
            raise Go2RtcError("camera source is not a go2rtc RTSP restream")
        return Go2RtcStream(parsed.hostname, stream_name)

    def snapshot(self, camera: CameraConfig, source: str = "live") -> bytes:
        stream = self.stream(camera, source)
        entry = self._streams(stream.host).get(stream.name, {})
        codecs = self._video_codecs(entry if isinstance(entry, dict) else {})
        if (
            not any(codec in {"H264", "AVC"} for codec in codecs)
            and any(codec in {"H265", "HEVC"} for codec in codecs)
        ):
            raise Go2RtcError("go2rtc JPEG snapshots are unavailable for H265 streams")
        image = self._snapshot_bytes(stream.host, stream.name)
        if not image.startswith(b"\xff\xd8"):
            raise Go2RtcError("go2rtc snapshot was not JPEG")
        return image

    def stream_info(self, camera: CameraConfig, source: str = "live") -> dict[str, Any]:
        stream = self.stream(camera, source)
        streams = self._streams(stream.host)
        entry = streams.get(stream.name) if isinstance(streams, dict) else None
        codecs = self._video_codecs(entry if isinstance(entry, dict) else {})
        codec = codecs[0] if codecs else ""
        return {
            "available": bool(entry is not None),
            "host": stream.host,
            "stream": stream.name,
            "video_codec": codec,
            "video_codecs": codecs,
            "compatibility": (
                "h264"
                if not any(item in {"H264", "AVC"} for item in codecs)
                and any(item in {"H265", "HEVC"} for item in codecs)
                else "native"
            ),
        }

    def websocket_url(self, camera: CameraConfig, source: str = "live", compatibility: str = "native") -> str:
        stream = self.stream(camera, source)
        stream_name = stream.name
        if compatibility == "h264":
            stream_name = self._ensure_h264(stream)
        return (
            f"ws://{self._host_authority(stream.host)}:{self.api_port}"
            f"/api/ws?src={quote(stream_name, safe='')}"
        )

    def status(self, cameras: list[CameraConfig]) -> dict[str, Any]:
        hosts: dict[str, dict[str, Any]] = {}
        for camera in cameras:
            try:
                stream = self.stream(camera, "live")
            except Go2RtcError:
                continue
            if stream.host in hosts:
                continue
            try:
                streams = self._streams(stream.host)
                hosts[stream.host] = {"host": stream.host, "healthy": True, "streams": len(streams), "error": ""}
            except Go2RtcError as exc:
                hosts[stream.host] = {"host": stream.host, "healthy": False, "streams": 0, "error": str(exc)[:160]}
        values = list(hosts.values())
        return {"healthy": bool(values) and all(item["healthy"] for item in values), "hosts": values}

    def invalidate(self, host: str) -> None:
        with self._lock:
            self._streams_cache.pop(host, None)

    def _ensure_h264(self, stream: Go2RtcStream) -> str:
        digest = hashlib.sha1(f"{stream.host}/{stream.name}".encode("utf-8")).hexdigest()[:8]
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stream.name).strip("_") or "stream"
        compat_name = f"survng_{stem}_{digest}_h264"
        lock_key = (stream.host, stream.name)
        with self._lock:
            compatibility_lock = self._compatibility_locks.setdefault(
                lock_key,
                threading.Lock(),
            )
        with compatibility_lock:
            streams = self._streams(stream.host)
            if compat_name in streams:
                return compat_name
            params = urlencode({
                "name": compat_name,
                "src": f"ffmpeg:{stream.name}#video=h264#width=1920",
            })
            request = Request(f"{self._base_url(stream.host)}/api/streams?{params}", method="PUT")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    if response.status >= 300:
                        raise Go2RtcError(f"go2rtc returned HTTP {response.status}")
            except OSError as exc:
                self.invalidate(stream.host)
                raise Go2RtcError(f"go2rtc compatibility stream failed: {exc}") from exc
            self.invalidate(stream.host)
        return compat_name

    def _streams(self, host: str, force: bool = False) -> dict[str, Any]:
        with self._lock:
            stream_lock = self._stream_locks.setdefault(host, threading.Lock())
        with stream_lock:
            return self._load_streams(host, force=force)

    def _load_streams(self, host: str, *, force: bool) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._streams_cache.get(host)
            if not force and cached is not None and cached[0] > now:
                return cached[1]
        try:
            with urlopen(f"{self._base_url(host)}/api/streams", timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, ValueError, TypeError) as exc:
            self.invalidate(host)
            raise Go2RtcError(f"go2rtc API unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            self.invalidate(host)
            raise Go2RtcError("go2rtc stream response was invalid")
        with self._lock:
            self._streams_cache[host] = (
                time.monotonic() + self.cache_seconds,
                payload,
            )
        return payload

    def _snapshot_bytes(self, host: str, stream_name: str) -> bytes:
        url = f"{self._base_url(host)}/api/frame.jpeg?{urlencode({'src': stream_name})}"
        try:
            with urlopen(url, timeout=max(self.timeout, 12.0)) as response:
                image = response.read(32 * 1024 * 1024 + 1)
                if len(image) > 32 * 1024 * 1024:
                    self.invalidate(host)
                    raise Go2RtcError("go2rtc snapshot exceeded the size limit")
                return image
        except OSError as exc:
            self.invalidate(host)
            raise Go2RtcError(f"go2rtc snapshot failed: {exc}") from exc

    @staticmethod
    def _video_codecs(entry: dict[str, Any]) -> list[str]:
        codecs: list[str] = []
        for producer in entry.get("producers") or []:
            if not isinstance(producer, dict):
                continue
            for media in producer.get("medias") or []:
                if not isinstance(media, str):
                    continue
                parts = [part.strip() for part in media.split(",")]
                if len(parts) < 3 or parts[0].lower() != "video":
                    continue
                for codec in parts[2:]:
                    normalized = codec.split("/")[0].upper()
                    if normalized and normalized not in codecs:
                        codecs.append(normalized)
        return codecs

    def _base_url(self, host: str) -> str:
        return f"http://{self._host_authority(host)}:{self.api_port}"

    @staticmethod
    def _host_authority(host: str) -> str:
        return f"[{host}]" if ":" in host and not host.startswith("[") else host
