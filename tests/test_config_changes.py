from __future__ import annotations

import asyncio
import os
import stat
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.config import AppConfig, CameraConfig, DetectionZone
from survng.app.config_changes import (
    MAX_ENTRIES,
    ConfigChangeActorMiddleware,
    ConfigChangeLog,
    actor_from_scope,
    describe_changes,
)
from survng.app.config_routes import ConfigRouteDependencies, create_config_router
from survng.app.security import required_api_scope


def _config(**camera_overrides) -> AppConfig:
    camera = {
        "id": "drive",
        "name": "Driveway",
        "stream_url": "rtsp://admin:camera-secret@10.0.0.4/main",
        **camera_overrides,
    }
    return AppConfig(
        mqtt={"host": "broker.local", "password": "old-secret"},
        cameras=[CameraConfig(**camera), CameraConfig(id="gate", name="Side Gate", stream_url="rtsp://gate/main")],
    )


def _labels(changes):
    return [(change["label"], change["before"], change["after"]) for change in changes]


def test_changes_are_labelled_by_camera_name_and_redacted() -> None:
    before = _config()
    after = before.model_copy(deep=True)
    after.cameras[0].stream_url = "rtsp://admin:new-secret@10.0.0.4/main?token=abc"
    after.cameras[0].record_sub = not before.cameras[0].record_sub
    after.cameras[0].zones = [DetectionZone(name="Street", points=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])]
    after.mqtt.password = "new-secret"
    after.mqtt.host = "broker.lan"
    after.web_auth.revoked_sessions = {"a" * 64: 2_000_000_000}

    changes = describe_changes(before.model_dump(mode="json"), after.model_dump(mode="json"))
    labels = _labels(changes)

    assert ("Driveway › Record sub", "Off", "On") in labels
    assert ("Driveway › Zones › Street", "not set", "added") in labels
    assert ("MQTT › Host", "broker.local", "broker.lan") in labels
    # A password change is recorded without either value.
    assert ("MQTT › Password", "hidden", "hidden") in labels
    url = next(change for change in changes if change["label"] == "Driveway › Stream URL")
    assert "camera-secret" not in url["before"] and "new-secret" not in url["after"] and "abc" not in url["after"]
    assert all(change["camera_id"] == "drive" for change in changes if change["label"].startswith("Driveway"))
    # Sign-outs rewrite revoked sessions; that is not a configuration change.
    assert not any("Revoked" in change["label"] for change in changes)
    flat = repr(changes)
    for secret in ("old-secret", "new-secret", "camera-secret"):
        assert secret not in flat


def test_added_removed_and_reordered_cameras() -> None:
    before = _config()
    after = before.model_copy(deep=True)
    after.cameras = [after.cameras[1], CameraConfig(id="porch", name="Porch", stream_url="rtsp://porch/main")]

    labels = _labels(describe_changes(before.model_dump(mode="json"), after.model_dump(mode="json")))

    assert ("Porch › Camera", "not set", "added") in labels
    assert ("Driveway › Camera", "present", "removed") in labels

    swapped = before.model_copy(deep=True)
    swapped.cameras = list(reversed(swapped.cameras))
    assert _labels(describe_changes(before.model_dump(mode="json"), swapped.model_dump(mode="json"))) == [
        ("Camera order", "previous order", "new order"),
    ]


def test_log_records_actor_bounds_entries_and_reads_newest_first(tmp_path) -> None:
    log = ConfigChangeLog(lambda: tmp_path / "config-changes.jsonl")
    first = _config()
    second = first.model_copy(deep=True)
    second.mqtt.host = "broker.lan"

    async def save_in_request() -> None:
        async def app(_scope, _receive, _send) -> None:
            log.record(first, second)

        principal = SimpleNamespace(kind="user", username="alex", name="Alex")
        await ConfigChangeActorMiddleware(app)({"type": "http", "survng_principal": principal}, None, None)

    asyncio.run(save_in_request())
    entry = log.recent()[0]
    assert entry["actor"] == "alex"
    assert _labels(entry["changes"]) == [("MQTT › Host", "broker.local", "broker.lan")]
    assert stat.S_IMODE(os.stat(tmp_path / "config-changes.jsonl").st_mode) == 0o600

    # Saving identical config records nothing; the file stays bounded.
    assert log.record(second, second) is None
    current = second
    for index in range(MAX_ENTRIES + 5):
        nxt = current.model_copy(deep=True)
        nxt.mqtt.host = f"broker-{index}"
        log.record(current, nxt)
        current = nxt
    lines = (tmp_path / "config-changes.jsonl").read_text().splitlines()
    assert len(lines) == MAX_ENTRIES
    newest = log.recent(2)
    assert [entry["changes"][0]["after"] for entry in newest] == [f"broker-{MAX_ENTRIES + 4}", f"broker-{MAX_ENTRIES + 3}"]
    assert newest[0]["actor"] == ""


def test_rolled_back_apply_is_recorded_as_a_second_change(tmp_path) -> None:
    log = ConfigChangeLog(lambda: tmp_path / "log.jsonl")
    current = _config()
    incoming = current.model_copy(deep=True)
    incoming.mqtt.host = "broker.lan"

    log.record(current, incoming)
    # Targeted apply failed and restored the previous file while the active
    # in-memory config never changed: the restore must still be visible.
    log.record(current, current)

    entries = log.recent()
    assert [_labels(entry["changes"]) for entry in entries] == [
        [("MQTT › Host", "broker.lan", "broker.local")],
        [("MQTT › Host", "broker.local", "broker.lan")],
    ]


def test_recording_failure_never_raises(tmp_path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    log = ConfigChangeLog(lambda: blocked / "log.jsonl")
    after = _config()
    after.mqtt.host = "elsewhere"

    assert log.record(_config(), after) is None
    assert log.recent() == []


def test_actor_names_users_and_api_tokens() -> None:
    assert actor_from_scope({}) == ""
    assert actor_from_scope({"survng_principal": SimpleNamespace(kind="user", username="sam", name="Sam")}) == "sam"
    assert actor_from_scope({"survng_principal": SimpleNamespace(kind="token", name="Home Assistant")}) == "API token Home Assistant"


def test_changes_route_is_admin_only_and_bounded() -> None:
    recent = Mock(return_value=[{"id": "x"}])
    router = create_config_router(ConfigRouteDependencies(
        get_config=lambda: _config(),
        get_manager=Mock(),
        publish_config=Mock(),
        apply_config=Mock(),
        reload_manager=Mock(),
        save_config=Mock(),
        validate_config=Mock(),
        lock=threading.RLock(),
        probe_limiter=threading.BoundedSemaphore(1),
        recent_changes=recent,
    ))
    endpoint = next(route.endpoint for route in router.routes if route.path == "/api/config/changes")

    assert endpoint(5) == {"changes": [{"id": "x"}]}
    endpoint(10_000)
    recent.assert_called_with(100)
    assert required_api_scope("GET", "/api/config/changes") == "admin"
    assert required_api_scope("GET", "/api/config") == "read"


def test_main_saves_record_history_with_the_actor_inside_the_security_boundary(monkeypatch) -> None:
    from survng.app import main

    previous = main.config
    incoming = previous.model_copy(deep=True)
    incoming.mqtt.host = "recorded.example"
    save = Mock()
    record = Mock()
    monkeypatch.setattr(main, "save_config", save)
    monkeypatch.setattr(main.CONFIG_CHANGE_LOG, "record", record)

    main.save_config_with_history(incoming, assign_ids=False)

    save.assert_called_once_with(incoming, assign_ids=False)
    record.assert_called_once_with(previous, incoming)
    # user_middleware lists the outermost first: the actor middleware must run
    # inside the security boundary, after the caller has been identified.
    order = [entry.cls for entry in main.app.user_middleware]
    assert order.index(ConfigChangeActorMiddleware) > order.index(main.SecurityBoundaryMiddleware)


def test_zone_point_edits_are_one_change_and_users_are_named() -> None:
    before = _config(zones=[{"name": "Street", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]}])
    before.web_auth.users = []
    after = before.model_copy(deep=True)
    after.cameras[0].zones[0].points[1].x = 0.5
    after.cameras[0].zones[0].points.append(after.cameras[0].zones[0].points[0].model_copy())

    labels = _labels(describe_changes(before.model_dump(mode="json"), after.model_dump(mode="json")))
    assert labels == [("Driveway › Zones › Street › Points", "3 items", "4 items")]

    users_before = {"web_auth": {"users": [{"id": "u1", "username": "alex", "role": "viewer"}]}}
    users_after = {"web_auth": {"users": [{"id": "u1", "username": "alex", "role": "admin"}]}}
    assert _labels(describe_changes(users_before, users_after)) == [("Web auth › Users › alex › Role", "viewer", "admin")]
