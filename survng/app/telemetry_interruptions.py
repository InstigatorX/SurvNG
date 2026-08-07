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
    observed_at: datetime | None = None,
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
    # Reserve an operator-facing outage marker for a sustained gap spanning
    # multiple expected samples. Known restart intervals are removed first so
    # a slow sampler resuming after startup remains independently visible.
    gap_threshold = max(150.0, float(expected_sample_seconds) * 2.5)
    restart_intervals = [
        (start_at, end_at)
        for item in interruptions
        if item.get("kind", "").endswith("_restart")
        and (start_at := _utc(item.get("start_at"))) is not None
        and (end_at := _utc(item.get("end_at"))) is not None
    ]

    def add_unexplained_gap(start_at: datetime, end_at: datetime) -> None:
        segments = [(start_at, end_at)]
        for restart_start, restart_end in restart_intervals:
            remaining: list[tuple[datetime, datetime]] = []
            for segment_start, segment_end in segments:
                if restart_end <= segment_start or restart_start >= segment_end:
                    remaining.append((segment_start, segment_end))
                    continue
                if restart_start > segment_start:
                    remaining.append((segment_start, restart_start))
                if restart_end < segment_end:
                    remaining.append((restart_end, segment_end))
            segments = remaining
        for segment_start, segment_end in segments:
            duration = (segment_end - segment_start).total_seconds()
            if duration <= gap_threshold:
                continue
            interruptions.append(
                {
                    "kind": "unknown_gap",
                    "start_at": segment_start.isoformat(),
                    "marker_at": segment_start.isoformat(),
                    "end_at": segment_end.isoformat(),
                    "duration_seconds": round(duration, 1),
                    "title": "Telemetry unavailable",
                    "description": "No lifecycle event explains this missing telemetry interval.",
                    "instance_id": "",
                }
            )

    for previous, current in zip(samples, samples[1:]):
        add_unexplained_gap(previous, current)

    observed = _utc(observed_at) if observed_at is not None else None
    if observed is not None:
        if samples and observed > samples[-1]:
            add_unexplained_gap(samples[-1], observed)
        elif not samples:
            latest_ready = next(
                (
                    event["at"]
                    for event in reversed(events)
                    if event.get("kind") == "startup_ready" and event["at"] < observed
                ),
                None,
            )
            if latest_ready is not None:
                add_unexplained_gap(latest_ready, observed)
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
    recent: list[tuple[dict[str, Any], datetime, datetime]] = []
    for item in interruptions:
        end_at = _utc(item.get("end_at"))
        if end_at is None:
            continue
        start_at = _utc(item.get("start_at"))
        if start_at is None:
            start_at = end_at - timedelta(
                seconds=max(0.0, float(item.get("duration_seconds") or 0.0))
            )
        if end_at >= cutoff and start_at <= current:
            recent.append((item, start_at, end_at))
    counts = {
        "controlled": sum(item.get("kind") == "controlled_restart" for item, _, _ in recent),
        "unexpected": sum(item.get("kind") == "unexpected_restart" for item, _, _ in recent),
        "unknown": sum(
            item.get("kind") in {"unknown_restart", "unknown_gap"}
            for item, _, _ in recent
        ),
    }
    return {
        "hours": max(1, hours),
        **counts,
        "total": len(recent),
        "duration_seconds": round(
            sum(
                max(
                    0.0,
                    (min(end_at, current) - max(start_at, cutoff)).total_seconds(),
                )
                for _, start_at, end_at in recent
            ),
            1,
        ),
    }
