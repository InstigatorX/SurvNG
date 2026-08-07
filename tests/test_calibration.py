from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from survng.app import main
from survng.app.calibration import (
    apply_calibration_changes,
    build_calibration_report,
    calibration_configuration_fingerprint,
    calibration_configuration_payload,
)
from survng.app.config import AppConfig, CameraConfig
from survng.app.events import EventStore
from survng.app.intelligence_routes import (
    CalibrationApplyRequest,
    CalibrationRollbackRequest,
    CalibrationRunRequest,
)


def test_calibration_fingerprint_excludes_camera_credentials() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://secret-user:secret-password@192.0.2.2/stream",
    )])

    payload = calibration_configuration_payload(config)
    encoded = str(payload)

    assert "secret-user" not in encoded
    assert "secret-password" not in encoded
    assert len(calibration_configuration_fingerprint(config)) == 64


def test_calibration_changes_apply_as_one_validated_candidate() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://192.0.2.2/stream",
    )])

    candidate, changes = apply_calibration_changes(config, [
        {
            "scope": "global",
            "setting": "motion.visual_backup_min_score",
            "proposed": 0.76,
        },
        {
            "scope": "camera",
            "camera_id": "gate",
            "setting": "motion.visual_backup_min_consecutive",
            "proposed": 4,
        },
        {
            "scope": "global",
            "setting": "detector.tracking.sample_fps",
            "proposed": 2.5,
        },
    ])

    assert config.motion_qualification.visual_backup_min_score == 0.70
    assert candidate.motion_qualification.visual_backup_min_score == 0.76
    assert candidate.cameras[0].motion_qualification.visual_backup_min_consecutive == 4
    assert candidate.detector.tracking.sample_fps == 2.5
    assert len(changes) == 3


def test_calibration_rejects_large_single_step_and_supports_inheritance() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://192.0.2.2/stream",
        motion_qualification={"visual_backup_min_score": 0.75},
    )])

    with pytest.raises(ValueError, match="at most 0.1"):
        apply_calibration_changes(config, [{
            "scope": "global",
            "setting": "motion.visual_backup_min_score",
            "proposed": 0.90,
        }])

    candidate, changes = apply_calibration_changes(config, [{
        "scope": "camera",
        "camera_id": "gate",
        "setting": "motion.visual_backup_min_score",
        "proposed": None,
    }])

    assert changes[0]["before"] == 0.75
    assert changes[0]["after"] is None
    assert candidate.cameras[0].motion_qualification.visual_backup_min_score is None


def test_system_report_keeps_dissenting_cameras_scoped_independently() -> None:
    config = AppConfig(cameras=[
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://192.0.2.2/a"),
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://192.0.2.3/a"),
        CameraConfig(id="porch", name="Porch", stream_url="rtsp://192.0.2.4/a"),
    ])
    common = {
        "setting": "visual_backup_min_score",
        "value": 0.75,
        "support_count": 3,
        "average_confidence": 0.9,
        "reasons": ["Repeated empty rescue attempts."],
        "evidence": [],
    }
    reports = {
        "gate": {"analyzed": 12, "recommendations": [common]},
        "yard": {"analyzed": 12, "recommendations": [common]},
        "porch": {
            "analyzed": 12,
            "recommendations": [{**common, "value": 0.65}],
        },
    }

    report = build_calibration_report(config, reports, mode="standard")

    assert not any(item["scope"] == "global" for item in report["recommendations"])
    assert {
        (item["camera_id"], item["proposed"])
        for item in report["recommendations"]
    } == {("gate", 0.75), ("yard", 0.75), ("porch", 0.65)}


def test_system_report_requires_all_cameras_for_global_recommendation() -> None:
    config = AppConfig(cameras=[
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://192.0.2.2/a"),
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://192.0.2.3/a"),
    ])
    common = {
        "setting": "visual_backup_min_score",
        "value": 0.75,
        "support_count": 3,
        "reasons": ["Repeated empty rescue attempts."],
    }

    partial = build_calibration_report(
        config,
        {"gate": {"analyzed": 12, "recommendations": [common]}},
        mode="standard",
    )
    complete = build_calibration_report(
        config,
        {
            "gate": {"analyzed": 12, "recommendations": [common]},
            "yard": {"analyzed": 12, "recommendations": [common]},
        },
        mode="standard",
    )

    assert not any(item["scope"] == "global" for item in partial["recommendations"])
    assert any(
        item["scope"] == "global" and item["proposed"] == 0.75
        for item in complete["recommendations"]
    )


def test_global_recommendation_preserves_needed_explicit_override_change() -> None:
    config = AppConfig(cameras=[
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://192.0.2.2/a"),
        CameraConfig(
            id="yard",
            name="Yard",
            stream_url="rtsp://192.0.2.3/a",
            motion_qualification={"visual_backup_min_score": 0.65},
        ),
    ])
    common = {
        "setting": "visual_backup_min_score",
        "value": 0.75,
        "support_count": 3,
        "reasons": ["Repeated empty rescue attempts."],
    }

    report = build_calibration_report(
        config,
        {
            "gate": {"analyzed": 12, "recommendations": [common]},
            "yard": {"analyzed": 12, "recommendations": [common]},
        },
        mode="standard",
    )

    assert any(item["scope"] == "global" for item in report["recommendations"])
    assert any(
        item["scope"] == "camera"
        and item["camera_id"] == "yard"
        and item["proposed"] == 0.75
        for item in report["recommendations"]
    )


def test_calibration_rejects_malformed_class_settings_and_non_finite_values() -> None:
    config = AppConfig()

    with pytest.raises(ValueError, match="valid object class"):
        apply_calibration_changes(config, [{
            "scope": "global",
            "setting": "detector.class_confidence.person.bad",
            "proposed": 0.7,
        }])
    with pytest.raises(ValueError, match="finite"):
        apply_calibration_changes(config, [{
            "scope": "global",
            "setting": "detector.tracking.sample_fps",
            "proposed": float("nan"),
        }])


def test_calibration_ledger_is_durable_and_contains_inverse_values() -> None:
    with TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        run = store.create_calibration_run(
            mode="quick",
            camera_ids=["gate"],
            configuration_fingerprint="a" * 64,
        )
        store.update_calibration_run(
            int(run["id"]),
            status="completed",
            result={"recommendations": []},
        )
        summary = store.calibration_runs()[0]
        detail = store.get_calibration_run(int(run["id"]))
        change_set = store.create_calibration_change_set(
            run_id=int(run["id"]),
            parent_change_set_id=None,
            action="apply",
            status="collecting",
            evaluation_hours=24,
            configuration_fingerprint_before="a" * 64,
            configuration_fingerprint_after="b" * 64,
            changes=[{
                "id": "1:1",
                "scope": "camera",
                "camera_id": "gate",
                "setting": "motion.visual_backup_min_score",
                "before": None,
                "after": 0.75,
            }],
            apply_result={"apply_mode": "hot"},
        )

        reloaded = EventStore(Path(tmpdir)).get_calibration_change_set(int(change_set["id"]))

        assert reloaded is not None
        assert summary["result"] == {}
        assert detail is not None and detail["result"] == {"recommendations": []}
        assert reloaded["changes"][0]["before"] is None
        assert reloaded["changes"][0]["after"] == 0.75
        assert reloaded["configuration_fingerprint_after"] == "b" * 64


def test_calibration_ledger_tracks_individually_rolled_back_changes() -> None:
    with TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        source = store.create_calibration_change_set(
            run_id=None,
            parent_change_set_id=None,
            action="apply",
            status="partially_rolled_back",
            evaluation_hours=24,
            configuration_fingerprint_before="a" * 64,
            configuration_fingerprint_after="b" * 64,
            changes=[],
            apply_result={},
        )
        store.create_calibration_change_set(
            run_id=None,
            parent_change_set_id=int(source["id"]),
            action="rollback",
            status="rolled_back",
            evaluation_hours=24,
            configuration_fingerprint_before="b" * 64,
            configuration_fingerprint_after="a" * 64,
            changes=[{"source_change_id": "source:1"}],
            apply_result={},
        )

        assert store.calibration_rollback_change_ids(int(source["id"])) == {"source:1"}


def test_calibration_apply_accepts_only_persisted_recommendation_ids() -> None:
    config = AppConfig(
        audit_ai={"allow_apply_recommendations": True},
        cameras=[CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://192.0.2.2/stream",
        )],
    )
    fingerprint = calibration_configuration_fingerprint(config)
    recommendation = {
        "id": "approved",
        "scope": "camera",
        "camera_id": "gate",
        "setting": "motion.visual_backup_min_score",
        "current": None,
        "current_effective": 0.70,
        "proposed": 0.76,
    }
    events = SimpleNamespace(
        get_calibration_run=lambda _run_id: {
            "id": 3,
            "status": "completed",
            "configuration_fingerprint": fingerprint,
            "result": {"recommendations": [recommendation]},
        },
        create_calibration_change_set=Mock(return_value={"id": 8, "status": "collecting"}),
    )
    request = CalibrationApplyRequest(
        recommendation_ids=["approved"],
        confirmed=True,
        configuration_fingerprint=fingerprint,
        evaluation_hours=24,
    )

    def apply(candidate: AppConfig):
        main.config = candidate
        return candidate, {
            "apply_mode": "hot",
            "camera_workers_restarted": False,
            "subsystems_restarted": [],
            "hot_updated": ["motion_qualification"],
        }

    with (
        patch.object(main, "config", config),
        patch.object(main, "manager", SimpleNamespace(events=events)),
        patch.object(main, "apply_config_update", side_effect=apply),
    ):
        response = main._intelligence_route_bundle.service.calibration_apply(3, request)

    assert response["ok"] is True
    persisted = events.create_calibration_change_set.call_args.kwargs["changes"]
    assert persisted[0]["before"] is None
    assert persisted[0]["after"] == 0.76


def test_calibration_run_records_worker_start_failure() -> None:
    config = AppConfig(
        audit_ai={"enabled": True, "api_key": "configured"},
        cameras=[CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://192.0.2.2/stream",
        )],
    )
    events = SimpleNamespace(
        calibration_runs=lambda _limit: [],
        calibration_change_sets=lambda _limit: [],
        create_calibration_run=Mock(return_value={"id": 11, "status": "queued"}),
        update_calibration_run=Mock(),
    )

    with (
        patch.object(main, "config", config),
        patch.object(main, "manager", SimpleNamespace(events=events)),
        patch("survng.app.main.threading.Thread.start", side_effect=RuntimeError("no threads")),
        pytest.raises(HTTPException) as raised,
    ):
        main._intelligence_route_bundle.service.start_calibration_run(CalibrationRunRequest(camera_ids=["gate"]))

    assert raised.value.status_code == 503
    assert events.update_calibration_run.call_args.kwargs["status"] == "failed"


def test_calibration_rollback_reports_newer_value_conflict() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://192.0.2.2/stream",
        motion_qualification={"visual_backup_min_score": 0.80},
    )])
    events = SimpleNamespace(
        calibration_rollback_change_ids=lambda _change_set_id: set(),
        get_calibration_change_set=lambda _change_set_id: {
        "id": 8,
        "run_id": 3,
        "action": "apply",
        "changes": [{
            "id": "3:1",
            "scope": "camera",
            "camera_id": "gate",
            "setting": "motion.visual_backup_min_score",
            "before": None,
            "after": 0.76,
        }],
    })
    request = CalibrationRollbackRequest(confirmed=True)

    with (
        patch.object(main, "config", config),
        patch.object(main, "manager", SimpleNamespace(events=events)),
        pytest.raises(HTTPException) as raised,
    ):
        main._intelligence_route_bundle.service.calibration_rollback(8, request)

    assert raised.value.status_code == 409
    assert raised.value.detail["conflicts"][0]["current"] == 0.80


def test_calibration_evaluation_refuses_confounded_configuration() -> None:
    baseline = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://192.0.2.2/stream",
    )])
    changed = baseline.model_copy(deep=True)
    changed.motion_qualification.visual_backup_min_score = 0.76
    events = SimpleNamespace(
        get_calibration_change_set=lambda _change_set_id: {
            "id": 8,
            "run_id": 3,
            "configuration_fingerprint_after": calibration_configuration_fingerprint(
                baseline
            ),
            "changes": [],
        },
        get_calibration_run=lambda _run_id: {"result": {}},
        update_calibration_evaluation=Mock(),
    )

    with patch.object(main, "config", changed):
        main._intelligence_route_bundle.service._run_calibration_evaluation(
            8,
            changed,
            SimpleNamespace(events=events),
        )

    evaluation = events.update_calibration_evaluation.call_args.args[1]
    assert evaluation["outcome"] == "inconclusive"
    assert evaluation["comparison_basis"] == "configuration_conflict"
