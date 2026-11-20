"""Regression coverage for remote encoder ownership and input boundaries."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request, Response
from pydantic import ValidationError

from survng.app import homekit_nodes
from survng.app.homekit import HomeKitService
from survng.app.homekit_routes import HomeKitRouteDependencies, NodeReport, create_homekit_router
from survng.app.media_sessions import MediaSessionManager
from survng.app.security import is_public_api_path, required_api_scope
from tests.test_homekit import configuration, _profile


@pytest.fixture
def remote(tmp_path):
    service = HomeKitService(configuration(tmp_path), MediaSessionManager(), "review", lambda _: True)
    enrollment = service.nodes.begin_enrollment("http://localhost")
    token = enrollment["command"].split("/enroll/")[1].split("'")[0]
    enrolled = service.nodes.complete_enrollment(token, "review")
    node = service.nodes.authenticate(enrolled["secret"])
    service.nodes.update_report(node["id"], {"hardware": "qsv", "profiles": {"porch": _profile()}, "backends": {"porch": "qsv"}})
    service.config.cameras[0].homekit.encoder = node["id"]
    router = create_homekit_router(HomeKitRouteDependencies(lambda: SimpleNamespace(homekit=service)))
    endpoints = {route.path: route.endpoint for route in router.routes}
    try:
        yield service, node, enrolled["secret"], endpoints
    finally:
        service.close()


def request(secret, chunks, before=None, headers=()):
    iterator = iter(enumerate(chunks))
    async def receive():
        index, chunk = next(iterator)
        if before:
            before(index)
        return {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
    return Request({"type": "http", "headers": [
        (b"authorization", f"Bearer {secret}".encode()), (b"x-survng-camera", b"porch"),
        (b"x-survng-tier", b"1"), *headers]}, receive)


def test_assigned_encoder_cannot_be_removed(remote):
    service, node, _, endpoints = remote
    with pytest.raises(HTTPException) as error:
        endpoints["/api/homekit/nodes/{node_id}"](node["id"], Response())
    assert error.value.status_code == 409
    assert node["id"] in service.nodes.nodes


def test_encoder_version_reports_update_status_independently_of_liveness(remote, monkeypatch):
    service, node, _, _ = remote
    available = "a" * 64
    release = Mock(return_value=(available, b"archive"))
    monkeypatch.setattr(homekit_nodes, "agent_release", release)

    assert service.status()["nodes"][0] == {
        "id": node["id"], "name": "review", "online": True, "hardware": "qsv",
        "lastSeen": int(node["lastSeen"]), "cameras": ["porch"],
        "runningVersion": None, "availableVersion": available, "updateState": "unknown",
    }
    for running, state in (("b" * 64, "outdated"), (available, "current")):
        service.nodes.update_report(node["id"], {"runningVersion": running})
        status = service.status()["nodes"][0]
        assert status["online"]
        assert status["runningVersion"] == running
        assert status["updateState"] == state
    node["lastSeen"] = 0
    assert service.status()["nodes"][0]["online"] is False
    assert service.status()["nodes"][0]["updateState"] == "current"
    release.side_effect = FileNotFoundError
    status = service.status()["nodes"][0]
    assert status["availableVersion"] is None
    assert status["updateState"] == "unavailable"


def test_resync_requests_are_scoped_and_retryable(remote, monkeypatch):
    service, node, secret, endpoints = remote
    available = "a" * 64
    release = Mock(return_value=(available, b"archive"))
    monkeypatch.setattr(homekit_nodes, "agent_release", release)
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(homekit_nodes.time, "monotonic", clock)
    real_sync = service.nodes.sync
    monkeypatch.setattr(service.nodes, "sync", lambda node, report, assign, timeout: real_sync(node, report, assign, 0))
    resync = endpoints["/api/homekit/nodes/{node_id}/resync"]
    sync = endpoints["/api/homekit/nodes/sync"]
    response = Response()
    node_request = request(secret, [])
    path = f"/api/homekit/nodes/{node['id']}/resync"
    assert required_api_scope("POST", path) == "admin"
    assert not is_public_api_path("POST", path)
    with pytest.raises(HTTPException) as error:
        resync("unknown", response)
    assert error.value.status_code == 404

    generation = service.nodes.generation
    assert resync(node["id"], response) == {"requested": True}
    assert response.headers["cache-control"] == "no-store"
    assert service.nodes.generation > generation
    first = sync(NodeReport(), node_request)["syncRequest"]
    assert len(first) == 32
    assert service.status()["nodes"][0]["updateState"] == "syncing"
    enrollment = service.nodes.begin_enrollment("http://localhost")
    token = enrollment["command"].split("/enroll/")[1].split("'")[0]
    other = service.nodes.complete_enrollment(token, "other")
    assert "syncRequest" not in sync(NodeReport(), request(other["secret"], []))
    assert resync(node["id"], response) == {"requested": True}
    assert sync(NodeReport(failedSyncRequest="0" * 32), node_request)["syncRequest"] == first
    assert "syncRequest" not in sync(NodeReport(failedSyncRequest=first), node_request)
    assert service.status()["nodes"][0]["updateState"] == "failed"

    assert resync(node["id"], response) == {"requested": True}
    second = sync(NodeReport(failedSyncRequest=first), node_request)["syncRequest"]
    assert second != first
    clock.return_value += 3600
    assert "syncRequest" not in sync(NodeReport(), node_request)
    assert service.status()["nodes"][0]["updateState"] == "failed"
    assert resync(node["id"], response) == {"requested": True}
    third = sync(NodeReport(), node_request)["syncRequest"]
    assert third != second

    available = "b" * 64
    release.return_value = (available, b"archive-b")
    assert "syncRequest" not in sync(NodeReport(), node_request)
    assert resync(node["id"], response) == {"requested": True}
    assert sync(NodeReport(), node_request)["syncRequest"] != third

    assert "syncRequest" not in sync(NodeReport(runningVersion=available), node_request)
    assert service.status()["nodes"][0]["updateState"] == "current"
    with pytest.raises(HTTPException) as error:
        resync(node["id"], response)
    assert error.value.status_code == 409
    node["lastSeen"] = 0
    with pytest.raises(HTTPException) as error:
        resync(node["id"], response)
    assert error.value.status_code == 409
    service.nodes.update_report(node["id"], {})
    release.side_effect = FileNotFoundError
    with pytest.raises(HTTPException) as error:
        resync(node["id"], response)
    assert error.value.status_code == 409


def test_agent_archive_request_can_pin_release(remote, monkeypatch):
    _, _, secret, endpoints = remote
    available = "a" * 64
    monkeypatch.setattr(homekit_nodes, "agent_release", lambda: (available, b"archive"))
    download = endpoints["/api/homekit/nodes/agent"]
    node_request = request(secret, [])
    assert download(node_request).body == b"archive"
    assert download(node_request, release=available).body == b"archive"
    with pytest.raises(HTTPException) as error:
        download(node_request, release="b" * 64)
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        download(request("invalid", []), release=available)
    assert error.value.status_code == 401


def test_agent_release_digest_is_stable_across_cold_builds(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "dist/src").mkdir(parents=True)
    (tmp_path / "dist/src/agent.js").write_text("export {}")
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(homekit_nodes.time, "time", lambda: 100.0)
    first = homekit_nodes.agent_release(tmp_path)
    monkeypatch.setattr(homekit_nodes, "_agent_cache", None)
    monkeypatch.setattr(homekit_nodes.time, "time", lambda: 200.0)
    assert homekit_nodes.agent_release(tmp_path) == first
    (tmp_path / "dist/src/agent.js").write_text("export const changed = 1")
    assert homekit_nodes.agent_release(tmp_path)[0] != first[0]


@pytest.mark.parametrize("invalid", [{"runningVersion": "A" * 64}, {"failedSyncRequest": "g" * 32}, {"runningVersion": "1" * 65}])
def test_encoder_report_rejects_invalid_version_metadata(invalid):
    with pytest.raises(ValidationError):
        NodeReport(**invalid)


@pytest.mark.parametrize("change", ["reassign", "revoke", "power_off", "same_node_source"])
def test_media_upload_revalidates_authority_between_chunks(remote, change):
    service, node, secret, endpoints = remote
    feed = SimpleNamespace(wanted=lambda: True, write=lambda data: written.append(data), close=lambda: None)
    written = []
    service.nodes.feeds[("porch", 1)] = feed
    def before(index):
        if index != 1:
            return
        if change == "reassign":
            service.config.cameras[0].homekit.encoder = "local"
        elif change == "revoke":
            service.nodes.remove(node["id"], [])
        elif change == "power_off":
            service.camera_active = lambda _: False
        else:
            service.config.cameras[0].stream_url = "rtsp://replacement/main"

    revision = service.node_assignments()[0]["revision"]
    headers = [(b"x-survng-revision", revision.encode())]
    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoints["/api/homekit/nodes/media"](request(secret, [b"first", b"stale"], before, headers)))
    assert error.value.status_code == 404
    assert written == [b"first"]


def test_media_upload_requires_current_revision_at_reconnect(remote):
    service, _, secret, endpoints = remote
    written = []
    service.nodes.feeds[("porch", 1)] = SimpleNamespace(wanted=lambda: True, write=written.append, close=lambda: None)
    previous = service.node_assignments()[0]["revision"]
    service.config.cameras[0].stream_url = "rtsp://replacement/main"
    current = service.node_assignments()[0]["revision"]
    assert current != previous

    upload = endpoints["/api/homekit/nodes/media"]
    for headers in ([], [(b"x-survng-revision", previous.encode())]):
        with pytest.raises(HTTPException) as error:
            asyncio.run(upload(request(secret, [b"unread"], lambda _: pytest.fail("stale body read"), headers)))
        assert error.value.status_code == 404
    fresh = request(secret, [b"fresh"], headers=[(b"x-survng-revision", current.encode())])
    assert asyncio.run(upload(fresh)).status_code == 204
    assert written == [b"fresh"]


def test_snapshot_rejects_oversized_body_while_reading(remote):
    _, _, secret, endpoints = remote
    reads = []
    def before(index):
        reads.append(index)
    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoints["/api/homekit/nodes/snapshot"](request(secret, [b"x" * (4 * 1024 * 1024), b"x", b"unread"], before)))
    assert error.value.status_code == 413
    assert reads == [0, 1]


def test_profile_without_qualified_backend_is_not_ready(remote):
    service, node, _, _ = remote
    service.nodes.update_report(node["id"], {"hardware": "qsv", "profiles": {"porch": _profile()}})
    assert service.nodes.decision(service.config.cameras[0]) == "waiting"
    assert service.nodes.remote_for(service.config.cameras[0]) is None


def test_upload_claim_prevents_interleaved_or_stale_writers(remote):
    service, node, _, _ = remote
    written = []
    service.nodes.feeds[("porch", 1)] = SimpleNamespace(wanted=lambda: True, write=written.append, close=lambda: None)
    assignments = service.node_assignments()
    first = service.nodes.claim_feed(node, "porch", "1", assignments, assignments[0]["revision"])
    second = service.nodes.claim_feed(node, "porch", "1", assignments, assignments[0]["revision"])
    assert first is not None and second is not None
    assert not service.nodes.write_upload(node, "porch", "1", first, b"old", assignments)
    service.nodes.release_upload("porch", "1", first)
    assert service.nodes.write_upload(node, "porch", "1", second, b"new", assignments)
    assert written == [b"new"]


def test_late_snapshot_cannot_fulfil_a_new_request(remote, tmp_path):
    service, node, _, _ = remote
    destination = tmp_path / "snapshot.jpg"
    old = service.nodes.begin_snapshot("porch", destination)
    service.nodes.end_snapshot("porch", old)
    current = service.nodes.begin_snapshot("porch", destination)
    data = b"\xff\xd8jpeg"
    assignments = service.node_assignments()
    assert not service.nodes.accept_snapshot(node, "porch", data, assignments, old[2])
    assert not destination.exists()
    assert service.nodes.accept_snapshot(node, "porch", data, assignments, current[2])
    assert current[0].is_set() and destination.read_bytes() == data
    assert not service.nodes.accept_snapshot(node, "porch", b"\xff\xd8replacement", assignments, current[2])
    assert destination.read_bytes() == data


def test_pending_remote_snapshot_does_not_block_media_admission(remote):
    import io
    import json
    from survng.app.homekit import CameraWorker
    service, _, _, _ = remote
    worker = CameraWorker(service, service.config, service.config.cameras[0])
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker._admission({"version": 1, "id": 1, "method": "snapshot", "token": "snapshot"})
    assert worker.pending_snapshot is not None
    assert not worker.process.stdin.getvalue()
    worker._admission({"version": 1, "id": 2, "method": "acquire", "token": "relay", "purpose": "relay"})
    reply = json.loads(worker.process.stdin.getvalue())
    assert reply["id"] == 2 and reply["ok"]
    worker._admission({"version": 1, "id": 3, "method": "release", "token": "relay"})
    worker._poll_snapshot(cancel=True)
    assert worker.pending_snapshot is None
    assert service.nodes.snapshots == {}


def test_closed_registry_rejects_stale_sync_and_writers(remote):
    service, node, secret, _ = remote
    assignments = service.node_assignments()
    service.close()
    assert service.nodes.authenticate(secret) is None
    assert not service.nodes.feed_allowed(node, "porch", "1", assignments, assignments[0]["revision"])
    with pytest.raises((KeyError, RuntimeError)):
        service.nodes.sync(node, {}, assignments, timeout=0)
    with pytest.raises(RuntimeError):
        service.nodes.prepare_feeds("porch", [1])
    service.nodes.reopen()
    assert service.nodes.authenticate(secret) is not None


def test_registry_shutdown_retries_only_feeds_that_failed_to_close(remote):
    service, _, secret, _ = remote
    registry = service.nodes
    failed, other = Mock(), Mock()
    first_key, other_key = ("porch", 1), ("porch", 2)
    registry.feeds.update({first_key: failed, other_key: other})
    failed.close.side_effect = RuntimeError("HomeKit feed shutdown incomplete")

    with pytest.raises(RuntimeError, match="HomeKit feed shutdown incomplete"):
        registry.close()
    other.close.assert_called_once_with()
    assert registry.feeds == {first_key: failed}
    assert registry.authenticate(secret) is None
    with pytest.raises(RuntimeError, match="HomeKit feed shutdown incomplete"):
        registry.reopen()

    failed.close.side_effect = None
    registry.close()
    assert failed.close.call_count == 2
    other.close.assert_called_once_with()
    assert not registry.feeds
    registry.reopen()
    assert registry.authenticate(secret) is not None


def test_sync_evaluates_assignments_after_the_new_report(remote):
    service, node, _, _ = remote
    service.config.cameras[0].homekit.allow_software_encoding = False
    node["hardware"] = ""
    result = service.nodes.sync(node, {"hardware": "qsv"}, service.node_assignments, timeout=0)
    assert result["assignments"][0]["id"] == "porch"


@pytest.mark.parametrize("mutation", [
    lambda profile: profile["tiers"][0].update(width=100000),
    lambda profile: profile["tiers"].append(dict(profile["tiers"][0])),
    lambda profile: profile["tiers"][0].update(frameRate=0),
    lambda profile: profile["tiers"][0].update(peakBitrate=0),
    lambda profile: profile.update(resolutions=[[1920, 1080, 30]]),
    lambda profile: profile.update(resolutions=[]),
])
def test_invalid_remote_profiles_are_rejected(mutation):
    from survng.app.homekit_nodes import clean_profile
    profile = _profile()
    mutation(profile)
    assert clean_profile(profile) is None


def test_qualification_is_bound_to_the_exact_camera_assignment(remote):
    from survng.app.homekit_nodes import assignment_revision
    service, node, _, _ = remote
    camera = service.config.cameras[0]
    revision = assignment_revision(service.config, camera)
    report = {"hardware": "qsv", "profiles": {camera.id: _profile()}, "backends": {camera.id: "qsv"}}
    service.nodes.update_report(node["id"], report)
    assert service.nodes.decision(camera, revision) == "waiting"
    service.nodes.update_report(node["id"], {**report, "revisions": {camera.id: revision}})
    assert service.nodes.decision(camera, revision) == "remote"
    camera.stream_url = "rtsp://changed-source/live"
    changed = assignment_revision(service.config, camera)
    assert changed != revision
    assert service.nodes.remote_for(camera, changed) is None
    assert service.node_assignments()[0]["revision"] == changed


def test_installer_uses_private_temporary_files_and_valid_shell():
    import subprocess
    from survng.app.homekit_nodes import install_script
    script = install_script("https://survng.example", "a" * 40)
    assert 'work=$(mktemp -d)' in script
    assert '/tmp/survng-' not in script
    assert 'match the server architecture' in script
    assert 'Restart=always' in script
    result = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_remote_tools_resolve_the_configured_path_search(tmp_path, monkeypatch):
    from survng.app.homekit_nodes import ffmpeg_binaries
    for name in ("ffmpeg", "ffprobe"):
        path = tmp_path / name
        path.write_text("executable")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ffmpeg_binaries("ffmpeg") == (tmp_path / "ffmpeg", tmp_path / "ffprobe")


@pytest.mark.parametrize("state", ["not-json", '[]', '{"version":1,"nodes":null}', '{"version":1,"tokens":[null]}'])
def test_corrupt_optional_encoder_state_does_not_stop_survng(tmp_path, state):
    directory = tmp_path / "homekit"
    directory.mkdir(mode=0o700)
    saved = directory / "nodes.json"
    saved.write_text(state)
    saved.chmod(0o600)
    config = configuration(tmp_path)
    config.homekit.enabled = False
    service = HomeKitService(config, MediaSessionManager(), "review", lambda _: True)
    try:
        service.start()
        assert service.status()["nodeError"] == "node_state_unavailable"
        assert service.status()["nodes"] == []
        with pytest.raises(RuntimeError):
            service.nodes.begin_enrollment("http://localhost")
        assert saved.read_text() == state
    finally:
        service.close()


def test_failed_pairing_reset_restores_worker_supervision(remote, tmp_path):
    import fcntl
    from unittest.mock import Mock, patch
    service, _, _, _ = remote
    camera = service.config.cameras[0]
    directory = service.directory / camera.id
    directory.mkdir(mode=0o700)
    identity = directory / "identity.json"
    identity.write_text('preserve pairing')
    worker = Mock(path=directory, camera=camera)
    service.workers[camera.id] = worker
    with (directory / "owner.lock").open("w+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch("survng.app.homekit.CameraWorker") as factory:
            with pytest.raises(BlockingIOError):
                service.reset_pairing(camera.id)
            worker.close.assert_called_once()
            factory.return_value.start.assert_called_once()
            assert service.workers[camera.id] is factory.return_value
    assert identity.read_text() == 'preserve pairing'


@pytest.mark.parametrize("backend", [[], {}, None, 42])
def test_invalid_backend_report_fails_closed(remote, backend):
    service, node, _, _ = remote
    service.nodes.update_report(node["id"], {"profiles": {"porch": _profile()}, "backends": {"porch": backend}})
    assert service.nodes.decision(service.config.cameras[0]) == "waiting"
