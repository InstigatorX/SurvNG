"""Stable operational and diagnostic telemetry contracts.

Operational metrics are deliberately small and operator-facing.  Detailed
runtime dictionaries belong to bounded diagnostic sessions, never to the
always-on production history.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MetricKind(StrEnum):
    """How samples within an interval are summarized."""

    GAUGE = "gauge"
    COUNTER = "counter"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    kind: MetricKind
    unit: str


@dataclass(frozen=True, slots=True)
class TelemetryRetentionPolicy:
    raw_days: int = 2
    quarter_hour_days: int = 30
    hourly_days: int = 365
    operational_events_days: int = 90
    diagnostic_retention_days: int = 7
    operational_budget_bytes: int = 128 * 1024 * 1024
    diagnostic_budget_bytes: int = 256 * 1024 * 1024


SYSTEM_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("cpu_load_percent", MetricKind.GAUGE, "percent"),
    MetricDefinition("memory_used_percent", MetricKind.GAUGE, "percent"),
    MetricDefinition("application_rss_bytes", MetricKind.GAUGE, "bytes"),
    MetricDefinition("worker_rss_bytes", MetricKind.GAUGE, "bytes"),
    MetricDefinition("inference_ms", MetricKind.GAUGE, "milliseconds"),
    MetricDefinition("gpu_utilization_percent", MetricKind.GAUGE, "percent"),
    MetricDefinition("detector_requests", MetricKind.COUNTER, "count"),
    MetricDefinition("detector_failures", MetricKind.COUNTER, "count"),
    MetricDefinition("detector_capacity_delays", MetricKind.COUNTER, "count"),
    MetricDefinition("database_write_contention", MetricKind.COUNTER, "count"),
)


CAMERA_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("available", MetricKind.GAUGE, "boolean"),
    MetricDefinition("live_fps", MetricKind.GAUGE, "frames_per_second"),
    MetricDefinition("main_fps", MetricKind.GAUGE, "frames_per_second"),
    MetricDefinition("capture_interruptions", MetricKind.COUNTER, "count"),
    MetricDefinition("ema_frames_sampled", MetricKind.COUNTER, "count"),
    MetricDefinition("ema_frames_superseded", MetricKind.COUNTER, "count"),
    MetricDefinition("ema_credible_episodes", MetricKind.COUNTER, "count"),
    MetricDefinition("object_checks_admitted", MetricKind.COUNTER, "count"),
    MetricDefinition("object_checks_completed", MetricKind.COUNTER, "count"),
    MetricDefinition("object_check_failures", MetricKind.COUNTER, "count"),
    MetricDefinition("tracking_requested", MetricKind.COUNTER, "count"),
    MetricDefinition("tracking_completed", MetricKind.COUNTER, "count"),
    MetricDefinition("tracking_delayed", MetricKind.COUNTER, "count"),
    MetricDefinition("tracking_skipped", MetricKind.COUNTER, "count"),
    MetricDefinition("incidents_created", MetricKind.COUNTER, "count"),
)


DIAGNOSTIC_SCOPES: frozenset[str] = frozenset(
    {"system", "detector", "storage", "camera"}
)
DIAGNOSTIC_DURATIONS_SECONDS: tuple[int, ...] = (
    15 * 60,
    60 * 60,
    6 * 60 * 60,
    24 * 60 * 60,
)
