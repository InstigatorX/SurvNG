from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
import socket
import subprocess
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from survng.app.config import AppConfig, CameraConfig, CameraHomeKitConfig, HomeKitConfig
from survng.app.config_application import TargetedConfigApplication, manager_owned_config
from survng.app.homekit import CameraWorker, HomeKitService, camera_port, private_directory, worker_settings
from survng.app.homekit_nodes import agent_archive, ffmpeg_archive, install_script, media_ingest_root, node_releases, public_api_root, runtime_archive
from survng.app.homekit_routes import HomeKitRouteDependencies, PairingReset, create_homekit_router, pairing_qr
from survng.app.media_sessions import MediaSessionManager
from survng.app.security import is_public_api_path, required_api_scope
from fastapi import HTTPException, Response


def configuration(tmp_path):
    return AppConfig(storage_dir=str(tmp_path), homekit=HomeKitConfig(enabled=True, bind="192.168.1.2"), cameras=[
        CameraConfig(id="porch", name="Porch", stream_url="rtsp://user:private-password@camera/main",
            homekit=CameraHomeKitConfig(enabled=True, object_motion=True, allow_software_encoding=True)),
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://camera/yard", homekit=CameraHomeKitConfig(enabled=True, allow_software_encoding=True)),
    ])


def test_homekit_defaults_validation_and_stable_ports(tmp_path):
    assert not AppConfig().homekit.enabled
    assert not CameraHomeKitConfig().enabled
    assert not CameraHomeKitConfig().object_motion
    with pytest.raises(ValidationError):
        AppConfig(homekit={"enabled": True})
    with pytest.raises(ValidationError):
        HomeKitConfig(bind="0.0.0.0")
    with pytest.raises(ValidationError):
        CameraHomeKitConfig(port=80)
    config = configuration(tmp_path)
    assert camera_port(config.cameras[0]) == camera_port(config.cameras[::-1][1])
    config.cameras[1].homekit.port = camera_port(config.cameras[0])
    with pytest.raises(ValidationError, match="different ports"):
        AppConfig.model_validate(config.model_dump())


def test_motion_is_opt_in_recent_object_evidence_and_expires(tmp_path):
    config = configuration(tmp_path)
    active = Mock(return_value=True)
    service = HomeKitService(config, MediaSessionManager(), "generation", active)
    payload = {"camera_id": "porch", "timestamp": datetime.now(timezone.utc).isoformat(), "objects": [{"label": "person"}]}
    service.observe_objects(payload)
    assert service.motion_active("porch")
    service.motion.clear()
    for changes in ({"historical": True}, {"source": "manual_openvino"}, {"objects": []}, {"incident_objects": []},
                    {"objects": [{"label": "person", "incident_eligible": False}]},
                    {"timestamp": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()},
                    {"camera_id": "yard"}, {"timestamp": "invalid"}):
        service.observe_objects({**payload, **changes})
        assert not service.motion_active("porch")
    active.return_value = False
    service.observe_objects(payload)
    assert not service.motion_active("porch")
    assert "private-password" not in json.dumps(service.status())


def test_media_admission_releases_only_on_child_release(tmp_path):
    config = configuration(tmp_path)
    manager = MediaSessionManager()
    service = HomeKitService(config, manager, "generation", lambda _: True)
    worker = CameraWorker(service, config, config.cameras[0])
    worker.process = SimpleNamespace(pid=12345, stdin=io.BytesIO())
    worker._admission({"version": 1, "id": 1, "method": "acquire", "token": "1"})
    assert "1" in worker.leases
    lease = worker.leases["1"]
    manager.cancel_generation("generation", "stop")
    assert lease.cancelled()
    assert "1" in worker.leases  # cancellation is not cleanup
    worker._admission({"version": 1, "id": 2, "method": "release", "token": "1"})
    assert not worker.leases
    assert manager.wait_idle(0)


def test_two_camera_shared_sources_leave_capacity_for_viewers_and_snapshots(tmp_path):
    config = configuration(tmp_path)
    manager = MediaSessionManager()
    service = HomeKitService(config, manager, "generation", lambda _: True)
    workers = [CameraWorker(service, config, camera) for camera in config.cameras]
    for worker in workers:
        worker.process = SimpleNamespace(pid=12345, stdin=io.BytesIO())
        for token, purpose in (("1", "encoder"), ("2", "relay"), ("3", "relay"), ("4", "snapshot")):
            worker._admission({"version": 1, "id": token, "method": "acquire", "token": token, "purpose": purpose})
        assert len(worker.leases) == 4
    workers[0]._admission({"version": 1, "id": 5, "method": "acquire", "token": "5", "purpose": "encoder"})
    assert "5" not in workers[0].leases  # the two shared encoders fill the budget
    assert workers[0].status()["mediaCapacity"] == "unavailable"
    for worker in workers:
        for token in list(worker.leases):
            worker._admission({"version": 1, "id": token, "method": "release", "token": token})
    assert manager.wait_idle(0)


def test_per_camera_reconfiguration_preserves_other_workers_and_pairing(tmp_path):
    config = configuration(tmp_path)
    service = HomeKitService(config, MediaSessionManager(), "generation", lambda _: True)
    with patch("survng.app.homekit.CameraWorker") as factory:
        factory.side_effect = lambda service, app, camera: SimpleNamespace(camera=camera, start=Mock(), close=Mock(), status=lambda: {})
        service.start()
        first, other = service.workers.values()
        next_config = config.model_copy(deep=True)
        next_config.cameras[0].homekit.enabled = False
        service.reconfigure(next_config)
        first.close.assert_called_once()
        other.close.assert_not_called()
        assert service.workers["yard"] is other
        next_config = next_config.model_copy(deep=True)
        next_config.cameras[1].homekit.object_motion = True
        service.reconfigure(next_config)
        assert service.workers["yard"] is other
        other.close.assert_not_called()
        service.close()
        other.close.assert_called_once()


def test_config_homekit_is_targeted_not_camera_reload(tmp_path):
    current = configuration(tmp_path)
    incoming = current.model_copy(deep=True)
    incoming.cameras[0].homekit.object_motion = False
    assert manager_owned_config(current) == manager_owned_config(incoming)
    import threading
    runtime = Mock(config=current)
    application = TargetedConfigApplication(lock=threading.RLock(), save=Mock(), active_exports=lambda: [])
    _, result = application.apply(current, incoming, runtime, persist=False)
    runtime.homekit.reconfigure.assert_called_once_with(incoming)
    assert not result["camera_workers_restarted"]
    assert "homekit" in result["hot_updated"]


def test_private_state_rejects_permissions_and_symlinks(tmp_path):
    directory = tmp_path / "homekit"
    private_directory(directory)
    directory.chmod(0o755)
    with pytest.raises(RuntimeError):
        private_directory(directory)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(RuntimeError):
        private_directory(link)


def test_pairing_is_admin_only_not_cached_and_reset_requires_confirmation():
    service = Mock()
    service.pairing.return_value = {"setupCode": "123-45-679", "setupUri": "X-HM://0023ISYWY1234"}
    router = create_homekit_router(HomeKitRouteDependencies(lambda: SimpleNamespace(homekit=service)))
    pairing = next(route.endpoint for route in router.routes if route.path.endswith("/pairing"))
    response = Response()
    result = pairing("porch", response)
    assert response.headers["cache-control"] == "no-store"
    assert result["qrSvg"].startswith("<svg")
    assert "setupUri" not in result
    for path in ("/api/homekit/status", "/api/homekit/cameras/porch/pairing", "/api/homekit/cameras/porch/reset-pairing"):
        assert required_api_scope("GET", path) == "admin"
        assert required_api_scope("POST", path) == "admin"
    reset = next(route.endpoint for route in router.routes if route.path.endswith("/reset-pairing"))
    with pytest.raises(HTTPException):
        reset("porch", PairingReset(confirm_camera_id="yard"), Response())
    service.reset_pairing.assert_not_called()


def test_pairing_qr_round_trips_through_a_scanner():
    import cv2
    import numpy as np
    import re
    uri = "X-HM://0023ISYWY1234"
    svg = pairing_qr(uri)
    side = int(re.search(r'viewBox="0 0 (\d+)', svg)[1])
    image = np.full((side, side), 255, dtype=np.uint8)
    for x, y in re.findall(r'M(\d+) (\d+)h1', svg):
        image[int(y), int(x)] = 0
    image = np.repeat(np.repeat(image, 10, axis=0), 10, axis=1)
    assert cv2.QRCodeDetector().detectAndDecode(image)[0] == uri


def _profile():
    return {"width": 896, "height": 672, "tiers": [
        {"identifier": 1, "quality": 2, "width": 1600, "height": 1200, "frameRate": 30, "targetAverageBitrate": 1700, "peakBitrate": 1800},
    ], "resolutions": [[1600, 1200, 30], [1600, 1200, 15]]}


def test_encoder_enrollment_is_one_time_and_keeps_secrets_off_status(tmp_path):
    config = configuration(tmp_path)
    service = HomeKitService(config, MediaSessionManager(), "generation", lambda _: True)
    try:
        started = service.nodes.begin_enrollment("https://survng.example:8443")
        token = started["command"].split("/enroll/")[1].split("'")[0]
        script = service.nodes.enrollment_script(token)
        assert "ORIGIN='https://survng.example:8443'" in script
        assert f"TOKEN='{token}'" in script
        assert "linux-${node_arch}" in script
        assert "SURVNG_FFMPEG=%s" in script and "/opt/survng-homekit-node/bin/ffmpeg" in script
        assert "/api/homekit/nodes/ffmpeg" in script
        assert "apt-get install -y --no-install-recommends ca-certificates curl ffmpeg" not in script
        assert "private-password" not in script and "rtsp://" not in script
        enrolled = service.nodes.complete_enrollment(token, "gate box")
        with pytest.raises(KeyError):
            service.nodes.enrollment_script(token)
        with pytest.raises(KeyError):
            service.nodes.complete_enrollment(token, "gate box")
        secret = enrolled["secret"]
        assert secret not in (tmp_path / "homekit" / "nodes.json").read_text()
        node = service.nodes.authenticate(secret)
        assert node and service.nodes.authenticate("wrong") is None
        service.nodes.sync(node, {"name": "Gate box", "hardware": "qsv", "ffmpeg": "ffmpeg version",
            "profiles": {"porch": _profile()}, "backends": {"porch": "qsv"}, "fps": {"porch": 29.5}}, [], timeout=0)
        config.cameras[0].homekit.encoder = node["id"]
        visible = json.dumps(service.status())
        assert secret not in visible and token not in visible and "private-password" not in visible and "secretHash" not in visible
        assert "Gate box" in visible
        remote = service.nodes.remote_for(config.cameras[0])
        settings = worker_settings(config, config.cameras[0], remote)
        assert settings["remote"] is True and "url" not in settings
        assert "private-password" not in json.dumps(settings)
        assert worker_settings(config, config.cameras[1])["url"] == "rtsp://camera/yard"
        assert service.node_assignments()[0]["id"] == "porch"
        feed_dir = service.nodes.prepare_feeds("porch", [1])
        client = socket.socket(socket.AF_UNIX)
        client.connect(str(feed_dir / "1.sock"))
        deadline = time.monotonic() + 2
        while not service.nodes.feeds[("porch", 1)].wanted() and time.monotonic() < deadline:
            time.sleep(0.02)
        service.nodes.write_feed("porch", 1, b"mpegts")
        client.settimeout(1)
        assert client.recv(16) == b"mpegts"
        client.close()
        with service.nodes.lock:
            service.nodes.nodes[node["id"]]["lastSeen"] = 0
        assert service.nodes.decision(config.cameras[0]) == "offline"
        assert service.nodes.remote_for(config.cameras[0]) is None
        service.nodes.update_report(node["id"], {"hardware": "software", "backends": {"porch": "software"}, "profiles": {"porch": _profile()}})
        config.cameras[0].homekit.allow_software_encoding = False
        assert service.nodes.decision(config.cameras[0]) == "software_not_allowed"
        assert service.node_assignments() == []
        assert is_public_api_path("GET", f"/api/homekit/enroll/{token}")
        assert is_public_api_path("POST", "/api/homekit/nodes/sync")
        assert is_public_api_path("GET", "/api/homekit/nodes/agent")
        assert is_public_api_path("GET", "/api/homekit/nodes/ffmpeg")
        assert is_public_api_path("GET", "/api/homekit/nodes/runtime")
        assert not is_public_api_path("POST", "/api/homekit/nodes")
        assert required_api_scope("POST", "/api/homekit/nodes") == "admin"
        assert required_api_scope("DELETE", f"/api/homekit/nodes/{node['id']}") == "admin"
    finally:
        service.close()


def test_agent_archive_ships_the_program_without_cache(tmp_path):
    root = tmp_path / "worker"
    (root / "dist" / "src").mkdir(parents=True)
    (root / "dist" / "src" / "agent.js").write_text("export {}\n")
    (root / "node_modules" / ".cache").mkdir(parents=True)
    (root / "node_modules" / ".cache" / "secret").write_text("nope")
    (root / "node_modules" / "pkg").mkdir()
    (root / "node_modules" / "pkg" / "index.js").write_text("module\n")
    (root / "package.json").write_text("{}\n")
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(agent_archive(root)), mode="r:gz") as archive:
        names = archive.getnames()
    assert "dist/src/agent.js" in names
    assert "node_modules/pkg/index.js" in names
    assert all(".cache" not in name for name in names)


def test_encoder_assignment_restarts_only_that_camera(tmp_path):
    config = configuration(tmp_path)
    service = HomeKitService(config, MediaSessionManager(), "generation", lambda _: True)
    with patch("survng.app.homekit.CameraWorker") as factory:
        factory.side_effect = lambda service, app, camera: SimpleNamespace(camera=camera, start=Mock(), close=Mock(), status=lambda: {})
        service.start()
        first, other = service.workers["porch"], service.workers["yard"]
        next_config = config.model_copy(deep=True)
        next_config.cameras[0].homekit.encoder = "0123456789abcdef"
        service.reconfigure(next_config)
        first.close.assert_called_once()
        other.close.assert_not_called()
        service.close()


def test_ffmpeg_archive_ships_only_the_configured_tools(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "ffmpeg").write_bytes(b"ffmpeg-bytes")
    (tools / "ffprobe").write_bytes(b"ffprobe-bytes")
    (tools / "ffplay").write_bytes(b"not-shipped")
    with tarfile.open(fileobj=io.BytesIO(ffmpeg_archive(str(tools / "ffmpeg"))), mode="r:gz") as archive:
        assert archive.getnames() == ["ffmpeg", "ffprobe"]
    assert node_releases(str(tools / "ffmpeg"), tmp_path / "missing", lib_dir=tmp_path / "no-runtime") == {"ffmpeg": hashlib.sha256(b"ffmpeg-bytes" + b"ffprobe-bytes").hexdigest()}
    script = install_script("https://survng.example", "a" * 32)
    assert "/api/homekit/nodes/runtime" in script
    assert "LD_LIBRARY_PATH=/opt/survng-homekit-node/lib" in script
    assert "LIBVA_DRIVERS_PATH=/opt/survng-homekit-node/lib/dri" in script
    assert "libdrm2" in script and "intel-media-va-driver" not in script
    subprocess.run(["sh", "-n"], input=script, text=True, check=True)


def test_runtime_archive_ships_the_primary_va_stack(tmp_path):
    lib = tmp_path / "lib"
    (lib / "dri").mkdir(parents=True)
    for name in ("libva.so.2.2400.0", "libva-drm.so.2", "libigdgmm.so.12", "libmfx-gen.so.1.2", "libvpl.so.2"):
        (lib / name).write_bytes(name.encode())
    (lib / "libva.so.2").symlink_to("libva.so.2.2400.0")
    (lib / "dri" / "iHD_drv_video.so").write_bytes(b"ihd")
    (lib / "libva-x11.so.2").write_bytes(b"not-shipped")
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    (lib / "libevil.so.2").symlink_to(outside)
    with tarfile.open(fileobj=io.BytesIO(runtime_archive(lib)), mode="r:gz") as archive:
        assert archive.getnames() == ["libva.so.2", "libva-drm.so.2", "libigdgmm.so.12", "libmfx-gen.so.1.2", "libvpl.so.2", "dri/iHD_drv_video.so"]
        assert archive.extractfile("libva.so.2").read() == b"libva.so.2.2400.0"
    assert "runtime" in node_releases("/no/such/ffmpeg", tmp_path / "missing", lib_dir=lib)


def test_media_ingest_uses_the_lan_listener():
    assert media_ingest_root(("0.0.0.0", 8088), "192.168.86.243", "/survng") == "http://192.168.86.243:8088/survng"
    assert media_ingest_root(("192.168.86.243", 8088), "10.0.0.1", "") == "http://192.168.86.243:8088"


def test_installer_root_includes_the_survng_prefix():
    assert public_api_root("https://ha.example", "/survng") == "https://ha.example/survng"
    assert public_api_root("https://ha.example/survng/", "/survng") == "https://ha.example/survng"
    assert public_api_root("https://ha.example", "") == "https://ha.example"


def test_homekit_encoder_must_be_local_or_a_node_id():
    with pytest.raises(ValidationError):
        CameraHomeKitConfig(encoder="../other")


@pytest.mark.skipif(os.environ.get("HKSV_CAMERA_RUNTIME_TESTS") != "1", reason="optional Node 24 lifecycle integration")
def test_real_workers_keep_identity_reconfigure_independently_and_reset(tmp_path):
    def until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("HomeKit runtime condition timed out")
            time.sleep(0.05)

    config = configuration(tmp_path)
    # Isolated loopback publication only; the admin model requires a LAN address.
    config.homekit.bind = "127.0.0.1"
    config.homekit.node_path = os.environ["HKSV_NODE_PATH"]
    for camera in config.cameras:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            camera.homekit.port = reservation.getsockname()[1]
    media = MediaSessionManager()
    service = HomeKitService(config, media, "test-generation", lambda _: True)
    try:
        service.start()
        until(lambda: all(row["state"] == "running" for row in service.status()["cameras"]))
        first = service.workers["porch"]
        other_pid = service.workers["yard"].process.pid
        identity = json.loads((first.path / "identity.json").read_text())["id"]
        assert all(not row["paired"] for row in service.status()["cameras"])
        assert "private-password" not in json.dumps(service.status())
        next_config = config.model_copy(deep=True)
        next_config.cameras[0].homekit.enabled = False
        service.reconfigure(next_config)
        assert service.workers["yard"].process.pid == other_pid
        assert (first.path / "identity.json").is_file()
        service.reconfigure(config)
        until(lambda: service.workers["porch"].status()["state"] == "running")
        assert json.loads((first.path / "identity.json").read_text())["id"] == identity
        service.reset_pairing("porch")
        until(lambda: service.workers["porch"].status()["state"] == "running")
        assert json.loads((first.path / "identity.json").read_text())["id"] != identity
        assert service.workers["yard"].process.pid == other_pid
    finally:
        service.close()
    assert media.wait_idle(0)
    assert not list(service.directory.glob("*/control.sock"))
