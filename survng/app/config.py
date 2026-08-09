from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

CONFIG_PATH_ENV = "SURVNG_CONFIG_PATH"


class OnvifConfig(BaseModel):
    enabled: bool = False
    host: str = Field(default="", max_length=255)
    port: int = Field(default=8000, ge=1, le=65535)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)


class BaichuanConfig(BaseModel):
    enabled: bool = False
    host: str = Field(default="", max_length=255)
    port: int = Field(default=9000, ge=1, le=65535)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)
    channel: int = Field(default=0, ge=0, le=255)


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
    behavior: Literal["incident", "ignore", "none"] = "incident"
    exclude_from_ema: bool = False
    trigger: Literal["bottom_center"] = "bottom_center"


class MqttConfig(BaseModel):
    enabled: bool = False
    host: str = Field(default="", max_length=255)
    port: int = Field(default=1883, ge=1, le=65535)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)
    client_id: str = "survng"
    topic_prefix: str = "survng"
    qos: int = Field(default=0, ge=0, le=2)
    tls: bool = False
    discovery_enabled: bool = True
    discovery_prefix: str = "homeassistant"
    incident_events_enabled: bool = True
    server_status_enabled: bool = True
    server_name: str = Field(default="SurvNG Server", min_length=1, max_length=128)
    server_metrics_interval_seconds: int = Field(default=30, ge=10, le=3600)

    @field_validator("topic_prefix", "discovery_prefix", mode="before")
    @classmethod
    def normalize_mqtt_topic_prefix(cls, value: object, info: ValidationInfo) -> str:
        prefix = str(value or "").strip().strip("/")
        if not prefix:
            return "survng" if info.field_name == "topic_prefix" else "homeassistant"
        if any(character in prefix for character in ("+", "#", "\x00")):
            raise ValueError("MQTT topic prefix cannot contain wildcards or null bytes")
        return prefix


class AuditAiConfig(BaseModel):
    enabled: bool = False
    assistant_enabled: bool = True
    provider: Literal["openai", "gemini", "openai_compatible"] = "openai"
    api_key: str = Field(default="", max_length=4096)
    base_url: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=256)
    assistant_reasoning_model: str = Field(default="", max_length=256)
    timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)
    allow_apply_recommendations: bool = False


class SemanticSearchConfig(BaseModel):
    """Optional local vision-language search runtime and indexing policy."""

    enabled: bool = False
    implementation: Literal["mobileclip2_openvino", "openvino_manifest"] = (
        "mobileclip2_openvino"
    )
    model_dir: str = Field(default="", max_length=4096)
    device: str = Field(default="GPU", min_length=1, max_length=64)
    index_full_frame: bool = True
    index_object_crops: bool = True
    max_object_crops_per_event: int = Field(default=24, ge=1, le=100)
    max_results: int = Field(default=100, ge=10, le=500)
    backfill_batch_size: int = Field(default=25, ge=1, le=250)
    backfill_pause_seconds: float = Field(default=0.25, ge=0.01, le=5.0)
    worker_queue_size: int = Field(default=256, ge=16, le=4096)

    @field_validator("model_dir", mode="before")
    @classmethod
    def normalize_model_dir(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device(cls, value: object) -> str:
        return str(value or "GPU").strip().upper() or "GPU"


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
    # Legacy off/audit/enforce values remain loadable for backward
    # compatibility; the GUI emits camera/camera_rescue/adaptive for new saves.
    mode: Literal["camera", "camera_rescue", "adaptive", "off", "audit", "enforce"] = "camera"
    sensitivity: Literal["low", "balanced", "high"] = "balanced"
    stationary_object_tolerance: Literal["low", "balanced", "high"] = "balanced"
    illumination_filter_enabled: bool = False
    frame_width: int = Field(default=320, ge=240, le=960)
    sample_fps: float = Field(default=5.0, ge=2.0, le=10.0)
    camera_mode_background_fps: float = Field(default=2.0, ge=0.5, le=5.0)
    visual_backup_warmup_seconds: float = Field(default=10.0, ge=0.0, le=120.0)
    visual_backup_grace_seconds: float = Field(default=1.5, ge=0.0, le=5.0)
    visual_backup_min_score: float = Field(default=0.70, ge=0.0, le=1.0)
    visual_backup_score_margin: float = Field(default=0.15, ge=0.0, le=0.5)
    visual_backup_min_consecutive: int = Field(default=3, ge=2, le=10)
    visual_backup_cooldown_seconds: float = Field(default=20.0, ge=5.0, le=300.0)
    visual_backup_max_triggers_5m: int = Field(default=3, ge=1, le=30)
    window_seconds: float = Field(default=1.6, ge=0.8, le=4.0)
    post_trigger_seconds: float = Field(default=2.5, ge=0.5, le=6.0)
    burst_quiet_seconds: float = Field(default=0.5, ge=0.1, le=2.0)
    rejected_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    suppression_verification_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    borderline_rescue_enabled: bool = True
    borderline_margin: float = Field(default=0.03, ge=0.0, le=0.10)
    mog2_audit_enabled: bool = False
    mog2_history_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    pipeline: MotionPipelineConfig = Field(default_factory=MotionPipelineConfig)


class CameraMotionQualificationConfig(BaseModel):
    mode: Literal["inherit", "camera", "camera_rescue", "adaptive", "off", "audit", "enforce"] = "inherit"
    sensitivity: Literal["inherit", "low", "balanced", "high"] = "inherit"
    stationary_object_tolerance: Literal["inherit", "low", "balanced", "high"] = "inherit"
    illumination_filter_enabled: bool | None = None
    frame_width: int | None = Field(default=None, ge=240, le=960)
    visual_backup_grace_seconds: float | None = Field(default=None, ge=0.0, le=5.0)
    visual_backup_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_backup_min_consecutive: int | None = Field(default=None, ge=2, le=10)
    visual_backup_cooldown_seconds: float | None = Field(default=None, ge=5.0, le=300.0)
    visual_backup_max_triggers_5m: int | None = Field(default=None, ge=1, le=30)
    borderline_rescue_enabled: bool | None = None
    borderline_margin: float | None = Field(default=None, ge=0.0, le=0.10)
    suppression_verification_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mog2_audit_enabled: bool | None = None
    pipeline: CameraMotionPipelineConfig = Field(default_factory=CameraMotionPipelineConfig)


class CameraRetentionConfig(BaseModel):
    main_days: int | None = Field(default=None, ge=1, le=3650)
    live_days: int | None = Field(default=None, ge=1, le=3650)


class RecordingRetentionConfig(BaseModel):
    enabled: bool = True
    automatic_cleanup: bool = False
    storage_limit_tb: float = Field(default=13.0, ge=0.1, le=1000.0)
    minimum_free_percent: float = Field(default=15.0, ge=1.0, le=95.0)
    target_free_percent: float = Field(default=20.0, ge=2.0, le=99.0)
    emergency_free_percent: float = Field(default=5.0, ge=0.5, le=50.0)
    main_days: int = Field(default=7, ge=1, le=3650)
    live_days: int = Field(default=21, ge=1, le=3650)
    cleanup_batch_files: int = Field(default=2000, ge=100, le=10000)

    @model_validator(mode="after")
    def validate_watermarks(self) -> "RecordingRetentionConfig":
        if self.target_free_percent <= self.minimum_free_percent:
            raise ValueError("retention target free percent must exceed the minimum")
        if self.emergency_free_percent >= self.minimum_free_percent:
            raise ValueError("retention emergency free percent must be below the minimum")
        return self


class ImageStorageConfig(BaseModel):
    format: Literal["webp", "jpeg"] = "webp"
    quality: int = Field(default=95, ge=1, le=100)


class CameraConfig(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    name: str = Field(min_length=1, max_length=128)
    video_backend: str = "url"
    stream_url: str = Field(max_length=4096)
    live_stream_url: str | None = Field(default=None, max_length=4096)
    record: bool = True
    record_sub: bool = False
    retention: CameraRetentionConfig = Field(default_factory=CameraRetentionConfig)
    require_incident_zone: bool | None = None
    object_activity_attribution: Literal["inherit", "off", "shadow", "enforce"] = "inherit"
    motion_qualification: CameraMotionQualificationConfig = Field(default_factory=CameraMotionQualificationConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    baichuan: BaichuanConfig = Field(default_factory=BaichuanConfig)
    zones: list[DetectionZone] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_connection_from_url(self) -> "CameraConfig":
        try:
            parsed = urlsplit(self.stream_url.strip())
        except ValueError as exc:
            raise ValueError("stream_url must be a valid camera URL") from exc
        if parsed.scheme.lower() not in {"rtsp", "rtsps", "reolink", "http", "https"}:
            raise ValueError("stream_url must use RTSP, Reolink, HTTP, or HTTPS")
        if not parsed.hostname:
            raise ValueError("stream_url must include a camera host")
        self.stream_url = self.stream_url.strip()
        if self.live_stream_url is not None:
            self.live_stream_url = self.live_stream_url.strip() or None
            if self.live_stream_url:
                try:
                    live = urlsplit(self.live_stream_url)
                except ValueError as exc:
                    raise ValueError("live_stream_url must be a valid camera URL") from exc
                if live.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not live.hostname:
                    raise ValueError("live_stream_url must include a supported scheme and camera host")
        apply_stream_url_defaults(self)
        return self

    def live_url(self) -> str:
        return self.live_stream_url or self.stream_url

    def normalized_source(self, source: str | None, default: str = "live") -> str:
        value = (source or default).lower()
        return "main" if value == "main" else "live"

    def source_url(self, source: str = "live") -> str:
        return self.stream_url if self.normalized_source(source) == "main" else self.live_url()



class CameraTransitionRoute(BaseModel):
    from_camera: str = Field(min_length=1, max_length=128)
    to_camera: str = Field(min_length=1, max_length=128)
    min_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    max_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    bidirectional: bool = False
    enabled: bool = True
    name: str = Field(default="", max_length=128)

    @model_validator(mode="after")
    def validate_route(self) -> "CameraTransitionRoute":
        self.from_camera = self.from_camera.strip()
        self.to_camera = self.to_camera.strip()
        self.name = self.name.strip()
        if self.from_camera == self.to_camera:
            raise ValueError("camera transition route must connect two different cameras")
        if self.max_seconds <= self.min_seconds:
            raise ValueError("camera transition route maximum time must exceed its minimum")
        return self


class ObjectTrackingConfig(BaseModel):
    enabled: bool = True
    implementation: str = Field(default="survng_hybrid", min_length=1, max_length=64)
    excluded_labels: list[str] = Field(default_factory=lambda: ["face"], max_length=128)
    sample_fps: float = Field(default=2.0, ge=0.5, le=5.0)
    max_session_seconds: float = Field(default=15.0, ge=3.0, le=120.0)
    lost_timeout_seconds: float = Field(default=3.0, ge=0.5, le=15.0)
    min_confirmations: int = Field(default=2, ge=1, le=10)
    low_confidence_threshold: float = Field(default=0.25, ge=0.01, le=0.95)
    match_iou_threshold: float = Field(default=0.20, ge=0.05, le=0.90)
    match_center_distance_ratio: float = Field(default=0.65, ge=0.1, le=2.0)
    max_active_cameras: int = Field(default=2, ge=1, le=16)
    adaptive_burst_enabled: bool = True
    burst_max_active_cameras: int = Field(default=3, ge=1, le=16)
    capacity_wait_seconds: float = Field(default=5.0, ge=0.0, le=30.0)
    deferred_reid_enabled: bool = True
    deferred_reid_delay_seconds: float = Field(default=20.0, ge=0.0, le=300.0)
    deferred_reid_min_crop_pixels: int = Field(default=4096, ge=256, le=10_000_000)
    deferred_reid_rate_per_minute: int = Field(default=6, ge=1, le=120)
    related_sequence_window_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    camera_transition_routes: list[CameraTransitionRoute] = Field(
        default_factory=list,
        max_length=256,
    )
    max_tracks_per_session: int = Field(default=100, ge=1, le=1000)
    reid_enabled: bool = False
    reid_model_path: str = Field(default="", max_length=4096)
    reid_device: str = Field(default="AUTO", min_length=1, max_length=64)
    reid_match_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    reid_max_age_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    reid_max_embeddings_per_frame: int = Field(default=8, ge=1, le=64)
    reid_refresh_interval_frames: int = Field(default=8, ge=1, le=120)
    vehicle_reid_enabled: bool = False
    vehicle_reid_model_path: str = Field(default="", max_length=4096)
    vehicle_reid_device: str = Field(default="AUTO", min_length=1, max_length=64)
    vehicle_reid_match_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    vehicle_reid_labels: list[str] = Field(
        default_factory=lambda: ["car", "truck", "bus", "motorcycle"],
        max_length=32,
    )

    @field_validator("implementation", mode="before")
    @classmethod
    def normalize_tracking_implementation(cls, value: object) -> str:
        implementation = str(value or "").strip().lower()
        if implementation in {
            "bytetrack",
            "ultralytics_botsort",
            "ultralytics_deepocsort",
        }:
            return "survng_hybrid"
        return implementation

    @model_validator(mode="after")
    def validate_reid_model(self) -> "ObjectTrackingConfig":
        if self.reid_enabled and not self.reid_model_path.strip():
            raise ValueError("reid_model_path is required when person ReID is enabled")
        if self.vehicle_reid_enabled and not self.vehicle_reid_model_path.strip():
            raise ValueError(
                "vehicle_reid_model_path is required when vehicle ReID is enabled"
            )
        self.vehicle_reid_labels = list(dict.fromkeys(
            label
            for value in self.vehicle_reid_labels
            if (label := str(value).strip().lower())
        ))
        self.excluded_labels = list(dict.fromkeys(
            label
            for value in self.excluded_labels
            if (label := str(value).strip().lower())
        ))
        if self.vehicle_reid_enabled and not self.vehicle_reid_labels:
            raise ValueError("vehicle_reid_labels must contain at least one label")
        if self.burst_max_active_cameras < self.max_active_cameras:
            raise ValueError(
                "burst_max_active_cameras cannot be lower than max_active_cameras"
            )
        return self

    def tracks_label(self, label: object) -> bool:
        normalized = str(label or "").strip().lower()
        return bool(normalized and normalized not in self.excluded_labels)

    @property
    def appearance_reid_enabled(self) -> bool:
        return self.reid_enabled or self.vehicle_reid_enabled

    def reid_enabled_for_label(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        if normalized == "person":
            return self.reid_enabled
        return self.vehicle_reid_enabled and normalized in self.vehicle_reid_labels

    def reid_threshold_for_label(self, label: str) -> float:
        return (
            self.reid_match_threshold
            if str(label or "").strip().lower() == "person"
            else self.vehicle_reid_match_threshold
        )


class DetectorConfig(BaseModel):
    enabled: bool = False
    backend: Literal["openvino", "coreml"] = "openvino"
    object_worker_count: int = Field(default=1, ge=1, le=4)
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
    face_embedding_model_path: str = Field(default="", max_length=4096)
    face_landmark_model_path: str = Field(default="", max_length=4096)
    face_detection_model_path: str = Field(default="", max_length=4096)
    face_recognition_device: str = Field(default="AUTO", min_length=1, max_length=64)
    face_detection_threshold: float = Field(default=0.60, ge=0.01, le=0.99)
    face_match_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    face_auto_identify_enabled: bool = False
    face_auto_identify_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    face_auto_identify_margin: float = Field(default=0.12, ge=0.0, le=1.0)
    face_min_size: int = Field(default=48, ge=16, le=1024)
    face_max_references: int = Field(default=20, ge=1, le=200)
    confidence_threshold: float = Field(default=0.45, ge=0.01, le=0.99)
    nms_threshold: float = Field(default=0.45, ge=0.01, le=0.99)
    event_confirmation_frames: int = Field(default=2, ge=1, le=5)
    event_class_confirmation_frames: dict[str, int] = Field(default_factory=dict)
    event_class_confidence_thresholds: dict[str, float] = Field(default_factory=dict)
    object_activity_attribution: Literal["off", "shadow", "enforce"] = "enforce"
    require_incident_zone: bool = True
    labels: list[str] = Field(default_factory=list)
    tracking: ObjectTrackingConfig = Field(default_factory=ObjectTrackingConfig)

    @field_validator("event_class_confirmation_frames", mode="before")
    @classmethod
    def normalize_event_class_confirmations(cls, value: object) -> dict[str, int]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("event class confirmations must be an object")
        if len(value) > 256:
            raise ValueError("event class confirmations cannot exceed 256 labels")
        normalized: dict[str, int] = {}
        for raw_label, raw_confirmations in value.items():
            label = str(raw_label or "").strip().lower()
            if not label:
                raise ValueError("event class confirmation labels cannot be empty")
            try:
                numeric_confirmations = float(raw_confirmations)
            except (TypeError, ValueError) as exc:
                raise ValueError("event class confirmations must be whole numbers") from exc
            if (
                isinstance(raw_confirmations, bool)
                or not numeric_confirmations.is_integer()
            ):
                raise ValueError("event class confirmations must be whole numbers")
            confirmations = int(numeric_confirmations)
            if confirmations < 1 or confirmations > 5:
                raise ValueError("event class confirmations must be between 1 and 5")
            normalized[label] = confirmations
        return normalized

    @field_validator("event_class_confidence_thresholds", mode="before")
    @classmethod
    def normalize_event_class_confidence_thresholds(
        cls,
        value: object,
    ) -> dict[str, float]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("event class confidence thresholds must be an object")
        if len(value) > 256:
            raise ValueError("event class confidence thresholds cannot exceed 256 labels")
        normalized: dict[str, float] = {}
        for raw_label, raw_threshold in value.items():
            label = str(raw_label or "").strip().lower()
            if not label:
                raise ValueError("event class confidence threshold labels cannot be empty")
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError("event class confidence thresholds must be numbers") from exc
            if isinstance(raw_threshold, bool) or not math.isfinite(threshold):
                raise ValueError("event class confidence thresholds must be finite numbers")
            if threshold < 0.01 or threshold > 0.99:
                raise ValueError("event class confidence thresholds must be between 0.01 and 0.99")
            normalized[label] = threshold
        return normalized

    def resolved_model_path(self) -> str:
        return self.model_path or self.model_xml

    def resolved_coreml_model_path(self) -> str:
        return self.coreml_model_path


class AppConfig(BaseModel):
    base_path: str = "/survng"
    storage_dir: str = "survng/storage"
    database_dir: str = ""
    recording_index_dir: str = ""
    ffmpeg_path: str = "ffmpeg"
    hardware_acceleration: str = "auto"
    event_clip_before_seconds: float = Field(default=5.0, ge=0.0, le=3600.0)
    event_clip_after_seconds: float = Field(default=5.0, ge=0.0, le=3600.0)
    incident_thumbnail_annotations: bool = False
    recording_segment_seconds: float = Field(default=10.0, ge=2.0, le=300.0)
    recording_cache_max_gb: float = Field(default=5.0, ge=0.5, le=100.0)
    recording_cache_max_days: int = Field(default=7, ge=1, le=90)
    recording_cache_prewarm: bool = True
    image_storage: ImageStorageConfig = Field(default_factory=ImageStorageConfig)
    retention: RecordingRetentionConfig = Field(default_factory=RecordingRetentionConfig)
    motion_qualification: MotionQualificationConfig = Field(default_factory=MotionQualificationConfig)
    audit_ai: AuditAiConfig = Field(default_factory=AuditAiConfig)
    semantic_search: SemanticSearchConfig = Field(default_factory=SemanticSearchConfig)
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
        normalized = f"/{path.strip('/')}"
        if (
            not re.fullmatch(r"/[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)*", normalized)
            or any(part in {".", ".."} for part in normalized.split("/"))
        ):
            raise ValueError("base_path contains unsupported or unsafe characters")
        return normalized

    @model_validator(mode="after")
    def validate_event_clip_window(self) -> "AppConfig":
        if self.event_clip_before_seconds + self.event_clip_after_seconds <= 0:
            raise ValueError("event clip window must include time before or after the event")
        camera_ids = [camera.id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera ids must be unique")
        known_camera_ids = set(camera_ids)
        route_edges: set[tuple[str, str]] = set()
        for route in self.detector.tracking.camera_transition_routes:
            unknown = {route.from_camera, route.to_camera} - known_camera_ids
            if unknown:
                raise ValueError(
                    "camera transition route references unknown camera(s): "
                    + ", ".join(sorted(unknown))
                )
            edges = {(route.from_camera, route.to_camera)}
            if route.bidirectional:
                edges.add((route.to_camera, route.from_camera))
            if edges & route_edges:
                raise ValueError("camera transition routes cannot contain duplicates")
            route_edges.update(edges)
        return self




def slugify_camera_id(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in (value or "").strip().lower():
        if char.isascii() and char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return ("".join(cleaned).strip("-") or "camera")[:128]


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
        camera.onvif = OnvifConfig.model_validate({
            **camera.onvif.model_dump(mode="json"),
            "host": camera.onvif.host or host,
            "username": camera.onvif.username or username,
            "password": camera.onvif.password or password,
        })

    if scheme == "reolink":
        query = parse_qs(parsed.query)
        channel_values = query.get("channel") or query.get("chn") or []
        try:
            channel = int(channel_values[0]) if channel_values else camera.baichuan.channel
        except (TypeError, ValueError) as exc:
            raise ValueError("Reolink channel must be a whole number") from exc
        if channel < 0 or channel > 255:
            raise ValueError("Reolink channel must be between 0 and 255")
        camera.video_backend = "baichuan_native"
        camera.baichuan = BaichuanConfig.model_validate({
            **camera.baichuan.model_dump(mode="json"),
            "enabled": True,
            "host": host,
            "port": parsed.port or camera.baichuan.port or 9000,
            "username": username or camera.baichuan.username,
            "password": password or camera.baichuan.password,
            "channel": channel,
        })
    elif camera.video_backend != "baichuan_native":
        # Non-Reolink URLs use their configured URL backend by default, but an
        # explicit native Baichuan selection is valid with an RTSP/HTTP URL
        # retained as the snapshot and detection fallback.
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
                suffix = f"-{index}"
                candidate = f"{base[:128 - len(suffix)]}{suffix}"
                index += 1
            camera.id = candidate
            used.add(candidate)
        apply_stream_url_defaults(camera)
    return config


def _config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    configured_path = os.environ.get(CONFIG_PATH_ENV, "").strip()
    return Path(configured_path or "config.json")


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = _config_path(path)
    if not config_path.exists():
        if path is None and os.environ.get(CONFIG_PATH_ENV, "").strip():
            raise FileNotFoundError(
                f"configured SurvNG config does not exist: {config_path}"
            )
        config_path = Path("config.example.json")
    with config_path.open("r", encoding="utf-8") as handle:
        return normalize_config(AppConfig.model_validate(json.load(handle)), assign_ids=False)


def save_config(
    config: AppConfig,
    path: str | Path | None = None,
    assign_ids: bool = True,
) -> None:
    config = normalize_config(config.model_copy(deep=True), assign_ids=assign_ids)
    config_path = _config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = 0o600
    try:
        existing_mode = config_path.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, existing_mode)
            json.dump(config.model_dump(mode="json"), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, config_path)
        temporary_path = None
        try:
            directory_fd = os.open(config_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def camera_by_id(config: AppConfig, camera_id: str) -> Optional[CameraConfig]:
    return next((camera for camera in config.cameras if camera.id == camera_id), None)
