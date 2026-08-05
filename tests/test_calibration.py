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


def test_system_report_prefers_global_consensus_and_keeps_exceptions() -> None:
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

    assert any(
        item["scope"] == "global"
        and item["setting"] == "motion.visual_backup_min_score"
        and item["proposed"] == 0.75
        for item in report["recommendations"]
    )
    assert any(
        item["scope"] == "camera"
        and item["camera_id"] == "porch"
        and item["proposed"] == 0.65
        for item in report["recommendations"]
    )


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
        assert reloaded["changes"][0]["before"] is None
        assert reloaded["changes"][0]["after"] == 0.75
        assert reloaded["configuration_fingerprint_after"] == "b" * 64


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
    request = main.CalibrationApplyRequest(
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
        response = main.calibration_apply(3, request)

    assert response["ok"] is True
    persisted = events.create_calibration_change_set.call_args.kwargs["changes"]
    assert persisted[0]["before"] is None
    assert persisted[0]["after"] == 0.76


def test_calibration_rollback_reports_newer_value_conflict() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://192.0.2.2/stream",
        motion_qualification={"visual_backup_min_score": 0.80},
    )])
    events = SimpleNamespace(get_calibration_change_set=lambda _change_set_id: {
        "id": 8,
        "run_id": 3,
        "changes": [{
            "id": "3:1",
            "scope": "camera",
            "camera_id": "gate",
            "setting": "motion.visual_backup_min_score",
            "before": None,
            "after": 0.76,
        }],
    })
    request = main.CalibrationRollbackRequest(confirmed=True)

    with (
        patch.object(main, "config", config),
        patch.object(main, "manager", SimpleNamespace(events=events)),
        pytest.raises(HTTPException) as raised,
    ):
        main.calibration_rollback(8, request)

    assert raised.value.status_code == 409
    assert raised.value.detail["conflicts"][0]["current"] == 0.80
