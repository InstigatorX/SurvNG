from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, Field, model_validator


class OnvifConfig(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 8000
    username: str = ""
    password: str = ""


class BaichuanConfig(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 9000
    username: str = ""
    password: str = ""
    channel: int = 0


class CameraConfig(BaseModel):
    id: str
    name: str
    video_backend: str = "url"
    stream_url: str
    live_stream_url: str | None = None
    record: bool = True
    record_sub: bool = False
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    baichuan: BaichuanConfig = Field(default_factory=BaichuanConfig)

    @model_validator(mode="after")
    def derive_connection_from_url(self) -> "CameraConfig":
        apply_stream_url_defaults(self)
        return self

    def live_url(self) -> str:
        return self.live_stream_url or self.stream_url

    def normalized_source(self, source: str | None, default: str = "live") -> str:
        value = (source or default).lower()
        return "main" if value == "main" else "live"

    def source_url(self, source: str = "live") -> str:
        return self.stream_url if self.normalized_source(source) == "main" else self.live_url()



class DetectorConfig(BaseModel):
    enabled: bool = False
    backend: str = "openvino"
    model_path: str = ""
    model_xml: str = ""
    coreml_model_path: str = ""
    labels_path: str = ""
    device: str = "CPU"
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    labels: list[str] = Field(default_factory=list)

    def resolved_model_path(self) -> str:
        return self.model_path or self.model_xml

    def resolved_coreml_model_path(self) -> str:
        return self.coreml_model_path


class AppConfig(BaseModel):
    storage_dir: str = "survng/storage"
    ffmpeg_path: str = "ffmpeg"
    hardware_acceleration: str = "auto"
    event_clip_before_seconds: float = 5.0
    event_clip_after_seconds: float = 5.0
    recording_segment_seconds: float = Field(default=10.0, ge=2.0, le=300.0)
    recording_cache_max_gb: float = Field(default=5.0, ge=0.5, le=100.0)
    recording_cache_max_days: int = Field(default=7, ge=1, le=90)
    recording_cache_prewarm: bool = True
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)




def slugify_camera_id(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in (value or "").strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-") or "camera"


def apply_stream_url_defaults(camera: CameraConfig) -> None:
    raw_url = camera.stream_url or camera.live_stream_url or ""
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return
    scheme = parsed.scheme.lower()
    if not scheme or not parsed.hostname:
        return

    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    host = parsed.hostname
    if host:
        camera.onvif.host = camera.onvif.host or host
        camera.onvif.username = camera.onvif.username or username
        camera.onvif.password = camera.onvif.password or password

    if scheme == "reolink":
        query = parse_qs(parsed.query)
        channel_values = query.get("channel") or query.get("chn") or []
        try:
            channel = int(channel_values[0]) if channel_values else camera.baichuan.channel
        except (TypeError, ValueError):
            channel = camera.baichuan.channel
        camera.video_backend = "baichuan_native"
        camera.baichuan.enabled = True
        camera.baichuan.host = host
        camera.baichuan.port = parsed.port or camera.baichuan.port or 9000
        camera.baichuan.username = username
        camera.baichuan.password = password
        camera.baichuan.channel = channel
    elif scheme == "rtsp":
        camera.video_backend = "url"
        camera.baichuan.enabled = False


def normalize_config(config: AppConfig, assign_ids: bool = False) -> AppConfig:
    used: set[str] = set()
    for camera in config.cameras:
        if assign_ids:
            base = slugify_camera_id(camera.name or camera.id)
            candidate = base
            index = 2
            while candidate in used:
                candidate = f"{base}-{index}"
                index += 1
            camera.id = candidate
            used.add(candidate)
        apply_stream_url_defaults(camera)
    return config


def load_config(path: str = "config.json") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path("config.example.json")
    with config_path.open("r", encoding="utf-8") as handle:
        return normalize_config(AppConfig.model_validate(json.load(handle)), assign_ids=False)


def save_config(config: AppConfig, path: str = "config.json") -> None:
    config = normalize_config(config, assign_ids=True)
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")


def camera_by_id(config: AppConfig, camera_id: str) -> Optional[CameraConfig]:
    return next((camera for camera in config.cameras if camera.id == camera_id), None)
