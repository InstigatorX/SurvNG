from datetime import datetime, timedelta, timezone
import io
import json
import os
import socket
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from survng.app.config import AppConfig, CameraConfig, CameraHomeKitConfig, HomeKitConfig
from survng.app.config_application import TargetedConfigApplication, manager_owned_config
from survng.app.homekit import CameraWorker, HomeKitService, camera_port, private_directory
from survng.app.homekit_routes import HomeKitRouteDependencies, PairingReset, create_homekit_router, pairing_qr
from survng.app.media_sessions import MediaSessionManager
from survng.app.security import required_api_scope
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
