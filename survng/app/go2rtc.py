from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import urlopen

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
            # Kept for older clients. SurvNG now always relays the configured
            # go2rtc stream without creating codec-conversion aliases.
            "compatibility": "native",
            "delivery": "native",
            "transcoding": False,
        }

    def audio_stream_info(self, camera: CameraConfig, source: str = "live") -> dict[str, Any]:
        stream = self.stream(camera, source)
        entry = self._stream_details(stream.host, stream.name)
        producers = entry.get("producers") or []
        media_ready = False
        for producer in producers:
            if not isinstance(producer, dict):
                continue
            for receiver in producer.get("receivers") or []:
                codec = receiver.get("codec") if isinstance(receiver, dict) else None
                if not isinstance(codec, dict):
                    continue
                media_ready = media_ready or codec.get("codec_type") in {"audio", "video"}
                if codec.get("codec_type") != "audio":
                    continue
                return {
                    "available": True,
                    "codec": str(codec.get("codec_name") or "").strip().lower(),
                    "sample_rate": int(codec.get("sample_rate") or 0),
                }
            for media in producer.get("medias") or []:
                if not isinstance(media, str):
                    continue
                parts = [part.strip() for part in media.split(",")]
                if len(parts) < 3 or parts[0].lower() not in {"audio", "video"}:
                    continue
                media_ready = True
                if parts[0].lower() != "audio":
                    continue
                encoding = parts[2].split("/", 1)
                codec = {
                    "mpeg4-generic": "aac",
                    "pcma": "pcm_alaw",
                    "pcmu": "pcm_mulaw",
                }.get(encoding[0].lower(), encoding[0].lower())
                try:
                    sample_rate = int(encoding[1].split("/", 1)[0]) if len(encoding) > 1 else 0
                except ValueError:
                    sample_rate = 0
                return {
                    "available": True,
                    "codec": codec,
                    "sample_rate": sample_rate,
                }
        return {
            "available": media_ready,
            "codec": "",
            "sample_rate": 0,
        }

    def websocket_url(self, camera: CameraConfig, source: str = "live") -> str:
        stream = self.stream(camera, source)
        return (
            f"ws://{self._host_authority(stream.host)}:{self.api_port}"
            f"/api/ws?src={quote(stream.name, safe='')}"
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

    def _stream_details(self, host: str, stream_name: str) -> dict[str, Any]:
        url = f"{self._base_url(host)}/api/streams?{urlencode({'src': stream_name})}"
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, ValueError, TypeError) as exc:
            raise Go2RtcError(f"go2rtc stream metadata unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise Go2RtcError("go2rtc stream metadata response was invalid")
        return payload

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
