"""Regression coverage for remote encoder ownership and input boundaries."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request, Response

from survng.app.homekit import HomeKitService
from survng.app.homekit_routes import HomeKitRouteDependencies, create_homekit_router
from survng.app.media_sessions import MediaSessionManager
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


@pytest.mark.parametrize("change", ["reassign", "revoke", "power_off"])
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
        else:
            service.camera_active = lambda _: False
    with pytest.raises(HTTPException):
        asyncio.run(endpoints["/api/homekit/nodes/media"](request(secret, [b"first", b"stale"], before)))
    assert written == [b"first"]


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
    first = service.nodes.claim_feed(node, "porch", "1", assignments)
    second = service.nodes.claim_feed(node, "porch", "1", assignments)
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
    assert not service.nodes.feed_allowed(node, "porch", "1", assignments)
    with pytest.raises((KeyError, RuntimeError)):
        service.nodes.sync(node, {}, assignments, timeout=0)
    with pytest.raises(RuntimeError):
        service.nodes.prepare_feeds("porch", [1])
    service.nodes.reopen()
    assert service.nodes.authenticate(secret) is not None


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
