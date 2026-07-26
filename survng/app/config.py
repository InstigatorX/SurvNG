from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ZonePoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class DetectionZone(BaseModel):
    name: str
    color: str = "#22c55e"
    enabled: bool = True
    points: list[ZonePoint] = Field(default_factory=list)
    object_classes: list[str] = Field(default_factory=list)
    confidence_threshold: float | None = Field(default=None, ge=0.01, le=0.99)
    behavior: Literal["incident", "ignore"] = "incident"
    trigger: Literal["bottom_center"] = "bottom_center"


class MqttConfig(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = Field(default=1883, ge=1, le=65535)
    username: str = ""
    password: str = ""
    client_id: str = "survng"
    topic_prefix: str = "survng"
    qos: int = Field(default=0, ge=0, le=2)
    tls: bool = False
    discovery_enabled: bool = True
    discovery_prefix: str = "homeassistant"
    incident_events_enabled: bool = True


class AuditAiConfig(BaseModel):
    enabled: bool = False
    provider: Literal["openai", "gemini", "openai_compatible"] = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)
    allow_apply_recommendations: bool = False


class MotionStageSelection(BaseModel):
    stage_id: str
    implementation: str
    options: dict[str, Any] = Field(default_factory=dict)
    parallel_group: str = ""

    @field_validator("stage_id", "implementation", mode="before")
    @classmethod
    def normalize_stage_name(cls, value: object) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("motion stage name cannot be empty")
        return normalized

    @field_validator("parallel_group", mode="before")
    @classmethod
    def normalize_parallel_group(cls, value: object) -> str:
        return str(value or "").strip()


def _validate_unique_motion_stage_ids(
    graphs: dict[str, list[MotionStageSelection] | None],
) -> None:
    for graph_name, stages in graphs.items():
        if stages is None:
            continue
        stage_ids = [stage.stage_id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"duplicate motion stage ID in {graph_name} graph")


class MotionPipelineConfig(BaseModel):
    qualification: list[MotionStageSelection] = Field(default_factory=list)
    observation: list[MotionStageSelection] = Field(default_factory=list)
    fusion: list[MotionStageSelection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_stage_ids(self) -> "MotionPipelineConfig":
        _validate_unique_motion_stage_ids(
            {
                "qualification": self.qualification,
                "observation": self.observation,
                "fusion": self.fusion,
            }
        )
        return self


class CameraMotionPipelineConfig(BaseModel):
    qualification: list[MotionStageSelection] | None = None
    observation: list[MotionStageSelection] | None = None
    fusion: list[MotionStageSelection] | None = None

    @model_validator(mode="after")
    def validate_unique_stage_ids(self) -> "CameraMotionPipelineConfig":
        _validate_unique_motion_stage_ids(
            {
                "qualification": self.qualification,
                "observation": self.observation,
                "fusion": self.fusion,
            }
        )
        return self


class MotionQualificationConfig(BaseModel):
    mode: Literal["off", "audit", "enforce"] = "audit"
    sensitivity: Literal["low", "balanced", "high"] = "balanced"
    frame_width: int = Field(default=320, ge=240, le=960)
    sample_fps: float = Field(default=5.0, ge=2.0, le=10.0)
    window_seconds: float = Field(default=1.6, ge=0.8, le=4.0)
    post_trigger_seconds: float = Field(default=2.5, ge=0.5, le=6.0)
    burst_quiet_seconds: float = Field(default=0.5, ge=0.1, le=2.0)
    rejected_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    borderline_rescue_enabled: bool = True
    borderline_margin: float = Field(default=0.03, ge=0.0, le=0.10)
    mog2_audit_enabled: bool = True
    mog2_history_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    pipeline: MotionPipelineConfig = Field(default_factory=MotionPipelineConfig)


class CameraMotionQualificationConfig(BaseModel):
    mode: Literal["inherit", "off", "audit", "enforce"] = "inherit"
    sensitivity: Literal["inherit", "low", "balanced", "high"] = "inherit"
    frame_width: int | None = Field(default=None, ge=240, le=960)
    borderline_rescue_enabled: bool | None = None
    borderline_margin: float | None = Field(default=None, ge=0.0, le=0.10)
    mog2_audit_enabled: bool | None = None
    pipeline: CameraMotionPipelineConfig = Field(default_factory=CameraMotionPipelineConfig)


class CameraConfig(BaseModel):
    id: str
    name: str
    video_backend: str = "url"
    stream_url: str
    live_stream_url: str | None = None
    record: bool = True
    record_sub: bool = False
    motion_qualification: CameraMotionQualificationConfig = Field(default_factory=CameraMotionQualificationConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    baichuan: BaichuanConfig = Field(default_factory=BaichuanConfig)
    zones: list[DetectionZone] = Field(default_factory=list)

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
    cache_enabled: bool = True
    cache_dir: str = ".cache/openvino"
    warmup_enabled: bool = True
    face_max_observations: int = Field(default=1000, ge=100, le=100000)
    face_recognition_enabled: bool = False
    face_embedding_model_path: str = ""
    face_landmark_model_path: str = ""
    face_recognition_device: str = "AUTO"
    face_match_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    face_min_size: int = Field(default=48, ge=16, le=1024)
    face_max_references: int = Field(default=20, ge=1, le=200)
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    labels: list[str] = Field(default_factory=list)

    def resolved_model_path(self) -> str:
        return self.model_path or self.model_xml

    def resolved_coreml_model_path(self) -> str:
        return self.coreml_model_path


class AppConfig(BaseModel):
    base_path: str = "/survng"
    storage_dir: str = "survng/storage"
    recording_index_dir: str = ""
    ffmpeg_path: str = "ffmpeg"
    hardware_acceleration: str = "auto"
    event_clip_before_seconds: float = 5.0
    event_clip_after_seconds: float = 5.0
    incident_thumbnail_annotations: bool = True
    recording_segment_seconds: float = Field(default=10.0, ge=2.0, le=300.0)
    recording_cache_max_gb: float = Field(default=5.0, ge=0.5, le=100.0)
    recording_cache_max_days: int = Field(default=7, ge=1, le=90)
    recording_cache_prewarm: bool = True
    motion_qualification: MotionQualificationConfig = Field(default_factory=MotionQualificationConfig)
    audit_ai: AuditAiConfig = Field(default_factory=AuditAiConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("base_path", mode="before")
    @classmethod
    def normalize_base_path(cls, value: object) -> str:
        path = str(value or "").strip()
        if not path or path == "/":
            return ""
        if "?" in path or "#" in path:
            raise ValueError("base_path must contain only a URL path")
        return f"/{path.strip('/')}"




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


def save_config(config: AppConfig, path: str = "config.json", assign_ids: bool = True) -> None:
    config = normalize_config(config, assign_ids=assign_ids)
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")


def camera_by_id(config: AppConfig, camera_id: str) -> Optional[CameraConfig]:
    return next((camera for camera in config.cameras if camera.id == camera_id), None)
