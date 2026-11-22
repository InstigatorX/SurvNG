"""Regression coverage for remote encoder ownership and input boundaries."""
import asyncio
import copy
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request, Response
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from survng.app import homekit_nodes
from survng.app.homekit import HomeKitService
from survng.app.homekit_routes import HomeKitRouteDependencies, NodeReport, create_homekit_router
from survng.app.media_sessions import MediaSessionManager
from survng.app.security import is_public_api_path, required_api_scope
from tests.test_homekit import configuration, _profile


@pytest.fixture
def remote(tmp_path):
    service = HomeKitService(configuration(tmp_path), MediaSessionManager(), "review", lambda _: True)
    service.nodes.reopen()
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


@pytest.fixture
def encoder_feed(remote):
    service, node, _, _ = remote
    directory = service.nodes.prepare_feeds("porch", [1])
    feed = service.nodes.feeds[("porch", 1)]
    clients = []

    def connect_reader(send_buffer=4096):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        clients.append(client)
        with feed.lock:
            previous = set(feed.clients)
        client.connect(str(directory / "1.sock"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with feed.lock:
                added = feed.clients - previous
            if added:
                peer = added.pop()
                peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer)
                client.setblocking(False)
                return client, peer
            time.sleep(0.01)
        pytest.fail("feed did not accept the viewer")

    try:
        yield service, node, feed, connect_reader
    finally:
        for client in clients:
            client.close()


def _read_available(client):
    received = bytearray()
    while True:
        try:
            part = client.recv(256 * 1024)
        except BlockingIOError:
            return received
        assert part, "feed disconnected a draining viewer"
        received.extend(part)


def test_paused_feed_reader_recovers_through_encoder_uploads(encoder_feed):
    service, node, feed, connect_reader = encoder_feed
    client, peer = connect_reader()
    revision = service.node_assignments()[0]["revision"]
    claim = service.claim_encoder_upload(node, "porch", "1", revision)
    assert claim is not None
    expected = bytearray(bytes(range(256)) * 2048)
    try:
        assert service.write_encoder_upload(node, "porch", "1", claim, bytes(expected))
        assert peer in feed.clients
        assert feed._outbound[peer], "paused viewer must retain its unsent bytes"
        received = bytearray()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            received.extend(_read_available(client))
            if len(received) == len(expected):
                break
            # Exercise production writes, rather than calling the private flush helper.
            chunk = bytes([len(expected) % 251]) * 188
            expected.extend(chunk)
            assert service.write_encoder_upload(node, "porch", "1", claim, chunk)
            time.sleep(0.001)
        assert received == expected
        assert peer in feed.clients
        assert not feed._outbound
    finally:
        service.release_encoder_upload("porch", "1", claim)
    assert not service.nodes.uploads
    feed.close()
    assert not feed.clients and not feed._outbound
    assert not feed.thread.is_alive()


def test_stalled_feed_viewer_does_not_disconnect_a_draining_viewer(encoder_feed):
    service, node, feed, connect_reader = encoder_feed
    _, stalled = connect_reader()
    client, healthy = connect_reader(send_buffer=1024 * 1024)
    revision = service.node_assignments()[0]["revision"]
    claim = service.claim_encoder_upload(node, "porch", "1", revision)
    assert claim is not None
    expected, received = bytearray(), bytearray()
    try:
        for index in range(80):
            chunk = bytes([index]) * (32 * 1024)
            expected.extend(chunk)
            assert service.write_encoder_upload(node, "porch", "1", claim, chunk)
            received.extend(_read_available(client))
            assert healthy in feed.clients
            if index == 0:
                assert stalled in feed.clients, "a short pause must not disconnect the viewer"
            assert len(feed._outbound.get(stalled, b"")) <= homekit_nodes.FEED_CLIENT_BUFFER
        assert stalled not in feed.clients
        assert stalled not in feed._outbound
        assert received == expected
        assert feed.wanted()
        # The surviving viewer continues on the same claim after the other is dropped.
        assert service.write_encoder_upload(node, "porch", "1", claim, b"still-live")
        assert _read_available(client) == b"still-live"
    finally:
        service.release_encoder_upload("porch", "1", claim)


def test_disconnected_feed_viewer_releases_pending_bytes_and_upload(encoder_feed):
    service, node, feed, connect_reader = encoder_feed
    client, peer = connect_reader()
    revision = service.node_assignments()[0]["revision"]
    claim = service.claim_encoder_upload(node, "porch", "1", revision)
    assert claim is not None
    try:
        assert service.write_encoder_upload(node, "porch", "1", claim, b"x" * (512 * 1024))
        assert peer in feed.clients
        assert feed._outbound[peer]
        client.close()
        assert not service.write_encoder_upload(node, "porch", "1", claim, b"unused")
        assert not feed.clients and not feed._outbound
    finally:
        service.release_encoder_upload("porch", "1", claim)
    assert not service.nodes.uploads


@pytest.mark.parametrize("camera_enabled", [True, False])
def test_assigned_encoder_cannot_be_removed(remote, camera_enabled):
    service, node, _, endpoints = remote
    service.config.cameras[0].homekit.enabled = camera_enabled
    with pytest.raises(HTTPException) as error:
        endpoints["/api/homekit/nodes/{node_id}"](node["id"], Response())
    assert error.value.status_code == 409
    assert node["id"] in service.nodes.nodes


def test_unassigned_encoder_can_be_removed(remote):
    service, node, secret, endpoints = remote
    service.config.cameras[0].homekit.encoder = "local"
    remove = endpoints["/api/homekit/nodes/{node_id}"]
    assert remove(node["id"], Response()) == {"removed": True}
    assert service.nodes.authenticate(secret) is None
    with pytest.raises(HTTPException) as error:
        remove(node["id"], Response())
    assert error.value.status_code == 404


def test_encoder_installer_requires_a_published_release(remote, monkeypatch):
    from survng.app import homekit_routes

    service, _, _, endpoints = remote
    monkeypatch.setattr(homekit_routes, "available_agent_version", lambda: None)
    enroll = Mock()
    monkeypatch.setattr(service.nodes, "begin_enrollment", enroll)
    with pytest.raises(HTTPException) as error:
        endpoints["/api/homekit/nodes"](request("admin", []), Response())
    assert error.value.status_code == 409
    enroll.assert_not_called()


def test_encoder_version_reports_update_status_independently_of_liveness(remote, monkeypatch):
    service, node, _, _ = remote
    available = "a" * 64
    release = Mock(return_value=available)
    monkeypatch.setattr(homekit_nodes, "available_agent_version", release)

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
    release.return_value = None
    status = service.status()["nodes"][0]
    assert status["availableVersion"] is None
    assert status["updateState"] == "unavailable"


def test_resync_requests_are_scoped_and_retryable(remote, monkeypatch):
    service, node, secret, endpoints = remote
    available = "a" * 64
    release = Mock(return_value=available)
    monkeypatch.setattr(homekit_nodes, "available_agent_version", release)
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
    release.return_value = available
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
    release.return_value = None
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


def test_unstamped_legacy_worker_is_not_offered(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "dist/src").mkdir(parents=True)
    (tmp_path / "dist/src/agent.js").write_text("export {}")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "dist/src/camera.js").write_text("export {}")
    assert homekit_nodes.available_agent_version(tmp_path) is None
    with pytest.raises(RuntimeError):
        homekit_nodes.agent_release(tmp_path)


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


def test_media_upload_allows_config_changes_while_waiting_for_body(remote):
    service, _, secret, endpoints = remote
    written = []
    service.nodes.feeds[("porch", 1)] = SimpleNamespace(wanted=lambda: True, write=written.append, close=lambda: None)
    revision = service.node_assignments()[0]["revision"]
    original = request(secret, [b"first", b"stale"], headers=[(b"x-survng-revision", revision.encode())])

    def change_assignment():
        assert service.lock.acquire(timeout=1), "upload held the service lock while waiting for a chunk"
        try:
            service.config.cameras[0].homekit.encoder = "local"
        finally:
            service.lock.release()

    async def receive():
        if written:
            await asyncio.to_thread(change_assignment)
        return await original.receive()

    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoints["/api/homekit/nodes/media"](Request(original.scope, receive)))
    assert error.value.status_code == 404
    assert written == [b"first"]
    assert not service.nodes.uploads


def test_media_upload_releases_claim_on_client_disconnect(remote):
    service, _, secret, endpoints = remote
    written = []
    service.nodes.feeds[("porch", 1)] = SimpleNamespace(wanted=lambda: True, write=written.append, close=lambda: None)
    revision = service.node_assignments()[0]["revision"]
    original = request(secret, [b"first", b"unused"], headers=[(b"x-survng-revision", revision.encode())])

    async def receive():
        if written:
            return {"type": "http.disconnect"}
        return await original.receive()

    with pytest.raises(ClientDisconnect):
        asyncio.run(endpoints["/api/homekit/nodes/media"](Request(original.scope, receive)))
    assert written == [b"first"]
    assert not service.nodes.uploads


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
    service, _, secret, endpoints = remote
    destination = tmp_path / "snapshot.jpg"
    old = service.nodes.begin_snapshot("porch", destination)
    service.nodes.end_snapshot("porch", old)
    current = service.nodes.begin_snapshot("porch", destination)
    data = b"\xff\xd8jpeg"
    endpoint = endpoints["/api/homekit/nodes/snapshot"]

    def send_snapshot(token, chunks):
        return asyncio.run(endpoint(request(secret, chunks, headers=[(b"x-survng-request", token.encode())])))

    with pytest.raises(HTTPException) as stale:
        send_snapshot(old[2], [data])
    assert stale.value.status_code == 404
    assert not destination.exists()
    assert send_snapshot(current[2], [data[:2], data[2:]]) == {"accepted": True}
    assert current[0].is_set() and destination.read_bytes() == data
    with pytest.raises(HTTPException) as duplicate:
        send_snapshot(current[2], [b"\xff\xd8replacement"])
    assert duplicate.value.status_code == 404
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


def test_staged_registry_loads_persisted_state_at_activation(tmp_path):
    directory = tmp_path / "homekit"
    previous = homekit_nodes.NodeRegistry(directory)
    staged = homekit_nodes.NodeRegistry(directory)
    assert not directory.exists()
    previous.reopen()
    enrollment = previous.begin_enrollment("http://localhost")
    token = enrollment["command"].split("/enroll/")[1].split("'")[0]
    enrolled = previous.complete_enrollment(token, "before handoff")
    previous.update_report(enrolled["id"], {"name": "at handoff"})
    previous.close()

    assert staged.authenticate(enrolled["secret"]) is None
    staged.reopen()
    try:
        assert staged.authenticate(enrolled["secret"])["name"] == "at handoff"
    finally:
        staged.close()


def test_closed_registry_preserves_enrollments_and_nodes(remote):
    service, node, _, _ = remote
    registry = service.nodes
    enrollment = registry.begin_enrollment("http://localhost")
    token = enrollment["command"].split("/enroll/")[1].split("'")[0]
    registry.close()
    nodes, tokens = copy.deepcopy(registry.nodes), copy.deepcopy(registry.tokens)
    saved = registry.path.read_bytes()
    for operation in (
        lambda: registry.begin_enrollment("http://localhost"),
        lambda: registry.enrollment_script(token, "https://example.org"),
        lambda: registry.complete_enrollment(token, "unused"),
        lambda: registry.remove(node["id"], []),
        lambda: registry.update_report(node["id"], {"name": "unused"}),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
        assert registry.tokens == tokens and registry.nodes == nodes
        assert registry.path.read_bytes() == saved


def test_registry_keeps_ownership_when_feed_startup_cleanup_fails(remote, monkeypatch):
    service, _, _, _ = remote
    registry = service.nodes
    path = service.directory / "feeds" / "porch" / "1.sock"
    original_unlink = type(path).unlink

    def unlink(file, *args, **kwargs):
        if file == path:
            raise OSError("injected socket unlink failure")
        return original_unlink(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("injected thread startup failure")))
        patch.setattr(type(path), "unlink", unlink)
        with pytest.raises(RuntimeError, match="startup cleanup incomplete"):
            registry.prepare_feeds("porch", [1])
    assert path.exists()
    assert registry.feeds[("porch", 1)].closed
    with pytest.raises(RuntimeError, match="shutdown incomplete"):
        registry.prepare_feeds("porch", [1])
    registry.close()
    assert not path.exists()
    registry.reopen()
    assert registry.prepare_feeds("porch", [1]) == path.parent


def test_registry_retains_only_unfinished_feeds_until_drained(remote, tmp_path):
    service, _, _, _ = remote
    registry = service.nodes
    entered, resume = threading.Event(), threading.Event()

    def delayed_close():
        entered.set()
        assert resume.wait(3)
        raise RuntimeError("incomplete test feed")

    pending, stopped = Mock(), Mock()
    pending.close.side_effect = delayed_close
    registry.feeds.update({("porch", 1): pending, ("porch", 2): stopped})
    waiter = registry.begin_snapshot("porch", tmp_path / "snapshot.jpg")
    with ThreadPoolExecutor(max_workers=1) as executor:
        draining = executor.submit(registry.close)
        try:
            assert entered.wait(2)
            assert registry.lock.acquire(timeout=1)
            try:
                assert registry.closed
                assert not registry.snapshots and not waiter[0].is_set()
                with pytest.raises(RuntimeError):
                    registry.reopen()
            finally:
                registry.lock.release()
        finally:
            resume.set()
        with pytest.raises(RuntimeError):
            draining.result(timeout=2)

    stopped.close.assert_called_once()
    with pytest.raises(RuntimeError):
        registry.reopen()
    with pytest.raises(RuntimeError):
        registry.prepare_feeds("porch", [1])
    registry.write_feed("porch", 1, b"unused")
    pending.write.assert_not_called()
    pending.close.side_effect = None
    registry.close()
    registry.close()
    assert pending.close.call_count == 2
    stopped.close.assert_called_once()
    registry.reopen()
    assert not registry.closed


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
    with pytest.raises(KeyError):
        service.nodes.sync(node, {"name": "from prior activation"}, assignments, timeout=0)
    assert service.nodes.authenticate(secret)["name"] == "review"


def test_closing_registry_wakes_sync_and_fences_its_reply_on_reopen(remote, monkeypatch):
    service, node, _, _ = remote
    registry = service.nodes
    waiting = threading.Event()
    original_wait = registry.updates.wait

    def wait(timeout):
        waiting.set()
        return original_wait(timeout)

    def assignments():
        registry.reopen()
        return []

    monkeypatch.setattr(registry.updates, "wait", wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(registry.sync, node, {}, assignments, 5)
        assert waiting.wait(2)
        registry.close()
        with pytest.raises(KeyError):
            pending.result(timeout=2)


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
    service.started = True
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
