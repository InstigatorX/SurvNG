from datetime import datetime, timedelta, timezone

from survng.app.system_review import (
    automatic_eligible,
    collect_stored_signals,
    daily_briefing,
    due_pass,
    rank_site_recommendations,
    review_progress,
    suggestion_class,
)


def test_tracking_sample_rate_waits_while_slots_are_full():
    recommendation = {
        "setting": "detector.tracking.sample_fps",
        "current": 2.0,
        "proposed": 2.5,
        "support_count": 4,
    }
    ranked = rank_site_recommendations([recommendation], {"tracking_saturated": True})
    assert ranked["recommendations"] == []
    assert any("slots" in note["text"] for note in ranked["notes"])


def test_confidence_is_not_loosened_to_explain_zone_policy():
    recommendation = {
        "setting": "detector.confidence_threshold",
        "current": 0.7,
        "proposed": 0.5,
        "support_count": 3,
    }
    ranked = rank_site_recommendations(
        [recommendation],
        {"establishment_reasons": {"ignored_zone": 4, "below_confidence": 2}},
    )
    assert ranked["recommendations"] == []
    assert suggestion_class(recommendation) is None
    notes = daily_briefing({"establishment_reasons": {"below_confidence": 2}, "grace_closures": 12})
    assert notes["recommendations"] == []
    assert all("45" not in note["text"] for note in notes["site_notes"])


def test_conflicting_values_keep_the_stronger_suggestion():
    weak = {"setting": "motion.visual_backup_cooldown_seconds", "camera_id": "gate", "proposed": 20, "current": 10, "support_count": 1}
    strong = {"setting": "motion.visual_backup_cooldown_seconds", "camera_id": "gate", "proposed": 30, "current": 10, "support_count": 4}
    ranked = rank_site_recommendations([weak, strong], {})
    assert ranked["recommendations"] == [strong]
    assert suggestion_class(strong) == "visual_backup"
    assert automatic_eligible(strong, {"visual_backup"})
    assert automatic_eligible(strong, {"visual_backup_cooldown"})
    assert not automatic_eligible(strong, set())
    assert suggestion_class({"setting": "motion.sensitivity", "proposed": "low", "current": "balanced"}) == "motion_sensitivity"
    assert suggestion_class({"setting": "motion.frame_width", "proposed": 480, "current": 640}) == "motion_analysis"
    assert suggestion_class({"setting": "detector.tracking.lost_timeout_seconds", "proposed": 4, "current": 2}) == "tracking"
    assert suggestion_class({"setting": "detector.require_incident_zone", "proposed": False, "current": True}) is None


def test_restore_inherit_is_its_own_automatic_class():
    recommendation = {"setting": "motion.sensitivity", "proposed": "inherit", "current": "high"}
    assert suggestion_class(recommendation) == "restore_inherit"


def test_weekly_pass_is_preferred_when_both_are_due():
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    assert due_pass(now, "off", None, None) is None
    assert due_pass(now, "weekly", None, None) == "weekly"
    assert due_pass(now, "daily", None, None) == "daily"
    assert due_pass(now, "weekly", now - timedelta(hours=1), now - timedelta(days=7)) == "weekly"
    assert due_pass(now, "weekly", now - timedelta(days=1), now - timedelta(hours=1)) is None
    assert due_pass(now, "weekly", now - timedelta(days=2), now - timedelta(days=2)) == "daily"
    assert due_pass(now, "weekly", now - timedelta(hours=1), now - timedelta(hours=1)) is None


def test_failed_system_run_does_not_advance_cadence() -> None:
    from survng.app.intelligence_routes import IntelligenceService

    runs = [
        {"mode": "system_weekly", "status": "failed", "created_at": "2026-09-28T00:00:00+00:00"},
        {"mode": "system_weekly", "status": "completed", "created_at": "2026-09-20T00:00:00+00:00",
         "completed_at": "2026-09-21T00:00:00+00:00"},
    ]
    assert IntelligenceService._latest_system_run(runs, "system_weekly") == datetime(2026, 9, 21, tzinfo=timezone.utc)
    assert IntelligenceService._latest_system_run(runs[:1], "system_weekly") is None


def test_failed_system_run_has_a_bounded_retry_backoff() -> None:
    from survng.app.intelligence_routes import IntelligenceService

    now = datetime(2026, 9, 28, 1, 0, tzinfo=timezone.utc)
    runs = [{
        "mode": "system_weekly",
        "status": "failed",
        "created_at": "2026-09-28T00:30:00+00:00",
    }]
    assert IntelligenceService._system_retry_blocked(
        runs, "system_weekly", now,
    )
    assert not IntelligenceService._system_retry_blocked(
        runs, "system_weekly", now + timedelta(minutes=31),
    )


def test_system_review_requires_explicit_opt_in() -> None:
    from survng.app.config import AppConfig

    assert AppConfig().system_review.cadence == "off"


def test_stored_signals_survive_a_database_without_scene_tables():
    import sqlite3

    connection = sqlite3.connect(":memory:")
    assert collect_stored_signals(connection, 0)["grace_closures"] == 0


def test_stored_signals_read_establishment_reasons_and_the_grace_tail():
    import tempfile
    from pathlib import Path

    from survng.app.events import EventStore

    with tempfile.TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        with store._connect() as connection:
            connection.execute(
                """
                insert into scene_activity_decisions(
                    id, camera_id, sample_ids_json, verdict, activity_epoch, reason,
                    policy_version, evidence_json, created_at
                ) values ('d1', 'gate', '[]', 'unsupported', 200, 'ignored_zone', '1', '{}', 200)
                """
            )
            connection.execute(
                """
                insert into scene_incidents(id, state, start_epoch, end_epoch, revision)
                values ('incident-1', 'complete', 100, 145, 1)
                """
            )
            connection.execute(
                """
                insert into scene_episodes(
                    id, incident_id, camera_id, start_epoch, end_epoch, last_activity_epoch, coverage_json
                ) values ('episode-1', 'incident-1', 'gate', 100, 145, 100, '{}')
                """
            )
            signals = collect_stored_signals(connection, 0)
        assert signals["establishment_reasons"]["ignored_zone"] == 1
        assert signals["grace_closures"] == 1
        assert signals["incidents_by_camera"] == [{"camera_id": "gate", "count": 1}]


def test_review_progress_names_the_camera_and_image():
    payload = review_progress(
        completed=1,
        total=9,
        phase="Reviewing images",
        camera_id="gate",
        camera_name="Front gate",
        images_done=1,
        images_total=4,
    )
    assert payload == {
        "completed": 1,
        "total": 9,
        "phase": "Reviewing images",
        "camera_id": "gate",
        "camera_name": "Front gate",
        "camera_index": 2,
        "images_done": 1,
        "images_total": 4,
    }


def test_unknown_automatic_class_is_rejected():
    from pydantic import ValidationError

    from survng.app.config import AppConfig

    AppConfig(system_review={"cadence": "weekly", "automatic_classes": ["restore_inherit"]})
    try:
        AppConfig(system_review={"automatic_classes": ["detector.confidence_threshold"]})
    except ValidationError:
        return
    raise AssertionError("confidence changes must stay out of the automatic allowlist")
