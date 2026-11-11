from datetime import datetime, timedelta, timezone

from survng.app.system_review import (
    automatic_eligible,
    collect_stored_signals,
    daily_briefing,
    due_pass,
    rank_site_recommendations,
    review_progress,
    suggestion_class,
    with_runtime_signals,
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
    notes = daily_briefing({"establishment_reasons": {"below_confidence": 2, "depth_ignore_zone": 3}})
    assert notes["recommendations"] == []
    assert "grace_closures" not in notes["signals"]
    assert any("depth limit" in note["text"] for note in notes["site_notes"])
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
    assert suggestion_class({"setting": "motion.frame_width", "proposed": 480, "current": 640, "scope": "camera"}) == "motion_analysis"
    assert suggestion_class({"setting": "motion.frame_width", "proposed": 480, "current": 640, "scope": "global"}) == "motion_analysis"
    assert suggestion_class({"setting": "detector.tracking.sample_fps", "proposed": 2.5, "current": 2, "scope": "camera"}) is None
    assert suggestion_class({"setting": "motion.sample_fps", "proposed": 6, "current": 5, "scope": "global"}) is None
    assert suggestion_class({"setting": "detector.tracking.sample_fps", "proposed": 2.5, "current": 2, "scope": "global"}) == "tracking"
    assert suggestion_class({"setting": "detector.tracking.lost_timeout_seconds", "proposed": 4, "current": 2, "scope": "global"}) is None
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
    assert collect_stored_signals(connection, 0)["establishment_reasons"] == {}


def test_one_queued_frame_is_not_treated_as_backlog():
    assert with_runtime_signals({}, tracking_limit=2, detector_queue=1)["detector_behind"] is False
    assert with_runtime_signals({}, tracking_limit=2, detector_queue=2)["detector_behind"] is True


def test_stored_signals_count_rejected_decisions_by_created_at():
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
                ) values
                    ('d1', 'gate', '[]', 'unsupported', null, 'ignored_zone', '1', '{}', 200),
                    ('d2', 'gate', '[]', 'unsupported', null, 'depth_ignore_zone', '1', '{}', 200),
                    ('d3', 'gate', '[]', 'unsupported', null, 'ignored_zone', '1', '{}', 10),
                    ('d4', 'gate', '[]', 'supported', 200, 'eligible_zone', '1', '{}', 200)
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
            signals = collect_stored_signals(connection, 100)
        assert signals["establishment_reasons"]["ignored_zone"] == 1
        assert signals["establishment_reasons"]["depth_ignore_zone"] == 1
        assert signals["establishment_reasons"]["eligible_zone"] == 1
        assert "grace_closures" not in signals
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


def _review_service(state: dict, store, *, apply_update=None):
    import threading
    from types import SimpleNamespace

    from survng.app.intelligence_routes import IntelligenceDependencies, IntelligenceService

    manager = SimpleNamespace(events=store)

    def default_apply(new_config, *_args, **_kwargs):
        state["config"] = new_config
        return new_config, {"apply_mode": "hot"}

    service = IntelligenceService(IntelligenceDependencies(
        get_config=lambda: state["config"],
        get_manager=lambda: manager,
        manager_lock=threading.RLock(),
        get_audit_ai_limiter=lambda: None,
        get_assistant_limiter=lambda: None,
        application_stopping=threading.Event(),
        incident_queries=None,
        system_telemetry=None,
        apply_config_update=apply_update or default_apply,
        begin_ai_operation=lambda _name: None,
        end_ai_operation=lambda _name: None,
        media_export_manager=lambda: None,
    ))
    return service, manager


def _evaluate_regressed_change(*, automatic: bool, fail_first_apply: bool = False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import Mock

    from survng.app.calibration import calibration_configuration_fingerprint
    from survng.app.config import AppConfig, CameraConfig
    from survng.app.events import EventStore

    applied = AppConfig(
        cameras=[CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://192.0.2.2/a",
            motion_qualification={"visual_backup_min_score": 0.75},
        )],
        system_review={"cadence": "weekly", "automatic_classes": ["visual_backup"]},
    )
    state = {"config": applied, "applies": 0}

    def apply_update(new_config, *_args, **_kwargs):
        state["applies"] += 1
        if fail_first_apply and state["applies"] == 1:
            raise RuntimeError("configuration change was not applied because AI work is active")
        state["config"] = new_config
        return new_config, {"apply_mode": "hot"}

    followup = {
        "analyzed": 4,
        "verdict_counts": {"likely_false_alarm": 4},
        "category_verdict_counts": {"visual_backup": {"likely_false_alarm": 4}},
    }
    with TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        run = store.create_calibration_run(
            mode="system_weekly",
            camera_ids=["gate"],
            configuration_fingerprint=calibration_configuration_fingerprint(applied),
        )
        store.update_calibration_run(int(run["id"]), status="completed", result={
            "camera_reports": {
                "gate": {
                    "analyzed": 4,
                    "verdict_counts": {},
                    "category_verdict_counts": {"visual_backup": {"consistent": 4}},
                },
            },
        })
        change_set = store.create_calibration_change_set(
            run_id=int(run["id"]),
            parent_change_set_id=None,
            action="apply",
            status="reviewing",
            evaluation_hours=24,
            configuration_fingerprint_before="a" * 64,
            configuration_fingerprint_after=calibration_configuration_fingerprint(applied),
            changes=[{
                "id": "1:1:motion.visual_backup_min_score:gate",
                "scope": "camera",
                "camera_id": "gate",
                "setting": "motion.visual_backup_min_score",
                "before": None,
                "after": 0.75,
            }],
            apply_result={"apply_mode": "hot", "automatic": automatic},
        )
        service, manager = _review_service(state, store, apply_update=apply_update)
        service._calibration_camera_review = Mock(return_value=followup)
        service._run_calibration_evaluation(int(change_set["id"]), applied, manager)
        stored = store.get_calibration_change_set(int(change_set["id"])) or {}
    return stored, state["config"]


def test_regressed_automatic_review_rolls_back_and_revokes_the_class() -> None:
    rolled, config = _evaluate_regressed_change(automatic=True)
    assert rolled["status"] == "rolled_back"
    assert rolled["evaluation"]["outcome"] == "regressed"
    assert rolled["evaluation"]["rolled_back"] is True
    assert config.cameras[0].motion_qualification.visual_backup_min_score is None
    assert config.system_review.automatic_classes == []


def test_regressed_manual_change_is_kept_but_its_class_is_turned_off() -> None:
    kept, config = _evaluate_regressed_change(automatic=False)
    assert kept["status"] == "evaluated"
    assert kept["evaluation"]["outcome"] == "regressed"
    assert not kept["evaluation"].get("rolled_back")
    assert config.cameras[0].motion_qualification.visual_backup_min_score == 0.75
    assert config.system_review.automatic_classes == []


def test_failed_automatic_rollback_is_recorded_and_still_turns_off_the_class() -> None:
    stored, config = _evaluate_regressed_change(automatic=True, fail_first_apply=True)
    assert stored["status"] == "evaluated"
    assert stored["evaluation"]["outcome"] == "regressed"
    assert stored["evaluation"]["rolled_back"] is False
    assert "AI work is active" in stored["evaluation"]["rollback_error"]
    assert config.cameras[0].motion_qualification.visual_backup_min_score == 0.75
    assert config.system_review.automatic_classes == []


def test_weekly_apply_stores_the_automatic_flag_with_the_change_set() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from survng.app.calibration import calibration_configuration_fingerprint
    from survng.app.config import AppConfig, CameraConfig
    from survng.app.events import EventStore

    config = AppConfig(
        audit_ai={"allow_apply_recommendations": True},
        cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://192.0.2.2/a")],
        system_review={"cadence": "weekly", "automatic_classes": ["visual_backup"]},
    )
    state = {"config": config}
    with TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp))
        run = store.create_calibration_run(
            mode="system_weekly",
            camera_ids=["gate"],
            configuration_fingerprint=calibration_configuration_fingerprint(config),
        )
        store.update_calibration_run(int(run["id"]), status="completed", result={
            "recommendations": [{
                "id": "r1",
                "scope": "camera",
                "camera_id": "gate",
                "setting": "motion.visual_backup_min_score",
                "current": None,
                "current_effective": 0.70,
                "proposed": 0.76,
            }],
        })
        service, manager = _review_service(state, store)
        service._apply_automatic_review(int(run["id"]), manager)
        change_sets = store.calibration_change_sets(10)
    assert len(change_sets) == 1
    assert change_sets[0]["status"] == "collecting"
    assert change_sets[0]["apply_result"]["automatic"] is True
    assert state["config"].cameras[0].motion_qualification.visual_backup_min_score == 0.76


def test_tracking_is_suggested_but_never_applied_automatically() -> None:
    from survng.app.config import AppConfig
    from survng.app.system_review import AUTOMATIC_CLASSES

    recommendation = {"setting": "detector.tracking.sample_fps", "proposed": 2.5, "current": 2, "scope": "global"}
    assert "tracking" not in AUTOMATIC_CLASSES
    assert not automatic_eligible(recommendation, {"tracking"})
    assert AppConfig(system_review={"automatic_classes": ["tracking", "visual_backup"]}).system_review.automatic_classes == ["visual_backup"]
    ranked = rank_site_recommendations([recommendation | {"id": "t"}], {})
    assert [item["id"] for item in ranked["recommendations"]] == ["t"]


def test_unknown_automatic_class_is_rejected():
    from pydantic import ValidationError

    from survng.app.config import AppConfig

    AppConfig(system_review={"cadence": "weekly", "automatic_classes": ["restore_inherit"]})
    try:
        AppConfig(system_review={"automatic_classes": ["detector.confidence_threshold"]})
    except ValidationError:
        return
    raise AssertionError("confidence changes must stay out of the automatic allowlist")
