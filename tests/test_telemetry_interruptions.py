from __future__ import annotations

from datetime import datetime, timezone

from survng.app.telemetry_interruptions import (
    classify_telemetry_interruptions,
    summarize_interruptions,
)


def event(instance: str, kind: str, at: str) -> dict:
    return {"instance_id": instance, "kind": kind, "occurred_at": at}


def test_clean_shutdown_classifies_controlled_restart() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00", "2026-08-07T10:02:00+00:00"],
        [
            event("old", "startup_ready", "2026-08-07T09:00:00+00:00"),
            event("old", "shutdown_requested", "2026-08-07T10:00:20+00:00"),
            event("old", "shutdown_completed", "2026-08-07T10:00:25+00:00"),
            event("new", "startup_started", "2026-08-07T10:00:30+00:00"),
            event("new", "startup_ready", "2026-08-07T10:00:40+00:00"),
        ],
    )

    assert len(interruptions) == 1
    assert interruptions[0]["kind"] == "controlled_restart"
    assert interruptions[0]["duration_seconds"] == 20.0


def test_missing_shutdown_classifies_unexpected_restart() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00", "2026-08-07T10:02:00+00:00"],
        [
            event("old", "startup_ready", "2026-08-07T09:00:00+00:00"),
            event("new", "startup_started", "2026-08-07T10:01:00+00:00"),
            event("new", "startup_ready", "2026-08-07T10:01:10+00:00"),
        ],
    )

    assert len(interruptions) == 1
    assert interruptions[0]["kind"] == "unexpected_restart"


def test_unexplained_sample_gap_remains_unknown() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00", "2026-08-07T10:03:00+00:00"],
        [],
    )

    assert interruptions[0]["kind"] == "unknown_gap"


def test_single_missed_minute_is_sampling_jitter_not_an_outage() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00", "2026-08-07T10:02:00+00:00"],
        [],
    )

    assert interruptions == []


def test_current_sampler_outage_is_visible_without_a_later_sample() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00"],
        [],
        observed_at=datetime(2026, 8, 7, 10, 4, tzinfo=timezone.utc),
    )

    assert len(interruptions) == 1
    assert interruptions[0]["kind"] == "unknown_gap"
    assert interruptions[0]["duration_seconds"] == 240.0


def test_ready_process_without_any_samples_reports_sampler_outage() -> None:
    interruptions = classify_telemetry_interruptions(
        [],
        [
            event("first", "startup_started", "2026-08-07T10:00:00+00:00"),
            event("first", "startup_ready", "2026-08-07T10:00:10+00:00"),
        ],
        observed_at=datetime(2026, 8, 7, 10, 4, tzinfo=timezone.utc),
    )

    assert len(interruptions) == 1
    assert interruptions[0]["kind"] == "unknown_gap"
    assert interruptions[0]["start_at"] == "2026-08-07T10:00:10+00:00"


def test_slow_sampling_after_restart_is_separate_from_controlled_restart() -> None:
    interruptions = classify_telemetry_interruptions(
        ["2026-08-07T10:00:00+00:00"],
        [
            event("old", "startup_ready", "2026-08-07T09:00:00+00:00"),
            event("old", "shutdown_requested", "2026-08-07T10:00:20+00:00"),
            event("old", "shutdown_completed", "2026-08-07T10:00:25+00:00"),
            event("new", "startup_started", "2026-08-07T10:00:30+00:00"),
            event("new", "startup_ready", "2026-08-07T10:00:40+00:00"),
        ],
        observed_at=datetime(2026, 8, 7, 10, 4, tzinfo=timezone.utc),
    )

    assert [item["kind"] for item in interruptions] == [
        "controlled_restart",
        "unknown_gap",
    ]
    assert interruptions[1]["start_at"] == "2026-08-07T10:00:40+00:00"


def test_first_lifecycle_baseline_is_not_reported_as_outage() -> None:
    interruptions = classify_telemetry_interruptions(
        [],
        [
            event("first", "startup_started", "2026-08-07T10:00:00+00:00"),
            event("first", "startup_ready", "2026-08-07T10:00:10+00:00"),
        ],
    )

    assert interruptions == []


def test_summary_counts_recent_classifications() -> None:
    summary = summarize_interruptions(
        [
            {
                "kind": "controlled_restart",
                "end_at": "2026-08-07T10:00:20+00:00",
                "duration_seconds": 20,
            },
            {
                "kind": "unknown_gap",
                "end_at": "2026-08-07T09:00:00+00:00",
                "duration_seconds": 120,
            },
        ],
        now=datetime(2026, 8, 7, 12, tzinfo=timezone.utc),
    )

    assert summary == {
        "hours": 24,
        "controlled": 1,
        "unexpected": 0,
        "unknown": 1,
        "total": 2,
        "duration_seconds": 140.0,
    }


def test_summary_clips_an_interruption_to_the_requested_window() -> None:
    summary = summarize_interruptions(
        [
            {
                "kind": "unknown_gap",
                "start_at": "2026-08-05T00:00:00+00:00",
                "end_at": "2026-08-07T11:00:00+00:00",
                "duration_seconds": 59 * 60 * 60,
            }
        ],
        now=datetime(2026, 8, 7, 12, tzinfo=timezone.utc),
        hours=24,
    )

    assert summary["unknown"] == 1
    assert summary["duration_seconds"] == 23 * 60 * 60
