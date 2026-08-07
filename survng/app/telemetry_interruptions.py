"""Classify telemetry interruptions from durable samples and lifecycle evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _utc(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_telemetry_interruptions(
    sample_times: list[str],
    lifecycle_events: list[dict[str, Any]],
    *,
    expected_sample_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Return proven restarts plus otherwise-unexplained telemetry gaps."""
    samples = sorted(item for value in sample_times if (item := _utc(value)) is not None)
    events = sorted(
        (
            {**event, "at": parsed}
            for event in lifecycle_events
            if (parsed := _utc(event.get("occurred_at"))) is not None
        ),
        key=lambda event: event["at"],
    )
    interruptions: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    for index, event in enumerate(events):
        if event.get("kind") != "startup_started":
            continue
        instance_id = str(event.get("instance_id") or "")
        if not instance_id or instance_id in seen_instances:
            continue
        seen_instances.add(instance_id)
        started_at = event["at"]
        ready_at = next(
            (
                candidate["at"]
                for candidate in events[index + 1 :]
                if candidate.get("instance_id") == instance_id
                and candidate.get("kind") == "startup_ready"
            ),
            started_at,
        )
        prior_events = events[:index]
        prior_terminal = prior_events[-1] if prior_events else None
        prior_samples = [sample for sample in samples if sample < started_at]
        if prior_terminal and prior_terminal.get("kind") == "shutdown_completed":
            shutdown_instance = str(prior_terminal.get("instance_id") or "")
            shutdown_requested = next(
                (
                    candidate["at"]
                    for candidate in reversed(prior_events)
                    if candidate.get("instance_id") == shutdown_instance
                    and candidate.get("kind") == "shutdown_requested"
                ),
                prior_terminal["at"],
            )
            kind = "controlled_restart"
            start_at = shutdown_requested
            title = "Controlled restart"
            description = "SurvNG stopped cleanly and started again."
        elif any(candidate.get("kind") == "startup_ready" for candidate in prior_events):
            kind = "unexpected_restart"
            start_at = prior_samples[-1] if prior_samples else started_at
            title = "Unexpected restart"
            description = "SurvNG started without a completed shutdown record."
        elif prior_samples:
            kind = "unknown_restart"
            start_at = prior_samples[-1]
            title = "Restart · cause unknown"
            description = "Lifecycle history is incomplete, so the restart cannot be classified."
        else:
            # The first lifecycle event in a new database is a baseline, not an outage.
            continue
        interruptions.append(
            {
                "kind": kind,
                "start_at": start_at.isoformat(),
                "marker_at": started_at.isoformat(),
                "end_at": max(started_at, ready_at).isoformat(),
                "duration_seconds": round(
                    max(0.0, (max(started_at, ready_at) - start_at).total_seconds()),
                    1,
                ),
                "title": title,
                "description": description,
                "instance_id": instance_id,
            }
        )

    # One skipped minute can result from normal scheduler or SQLite contention.
    # Keep that visible as a broken chart line, but reserve an operator-facing
    # outage marker for a sustained gap spanning multiple expected samples.
    gap_threshold = max(150.0, float(expected_sample_seconds) * 2.5)
    for previous, current in zip(samples, samples[1:]):
        if (current - previous).total_seconds() <= gap_threshold:
            continue
        overlaps_restart = any(
            _utc(item["start_at"]) <= current and _utc(item["end_at"]) >= previous
            for item in interruptions
        )
        if overlaps_restart:
            continue
        interruptions.append(
            {
                "kind": "unknown_gap",
                "start_at": previous.isoformat(),
                "marker_at": previous.isoformat(),
                "end_at": current.isoformat(),
                "duration_seconds": round((current - previous).total_seconds(), 1),
                "title": "Telemetry unavailable",
                "description": "No lifecycle event explains this missing telemetry interval.",
                "instance_id": "",
            }
        )
    return sorted(interruptions, key=lambda item: str(item["start_at"]))


def summarize_interruptions(
    interruptions: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current.astimezone(timezone.utc) - timedelta(hours=max(1, hours))
    recent = [
        item
        for item in interruptions
        if (end_at := _utc(item.get("end_at"))) is not None and end_at >= cutoff
    ]
    counts = {
        "controlled": sum(item.get("kind") == "controlled_restart" for item in recent),
        "unexpected": sum(item.get("kind") == "unexpected_restart" for item in recent),
        "unknown": sum(
            item.get("kind") in {"unknown_restart", "unknown_gap"} for item in recent
        ),
    }
    return {
        "hours": max(1, hours),
        **counts,
        "total": len(recent),
        "duration_seconds": round(
            sum(float(item.get("duration_seconds") or 0.0) for item in recent),
            1,
        ),
    }
