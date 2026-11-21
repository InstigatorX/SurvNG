"""Optional per-camera HKSV3 workers, owned by the active manager generation.

Only the supervisor pipe carries source credentials. The owner-only socket is
separate from observability; setup material is returned only by admin routes.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import select
import signal
import shutil
import socket
import stat
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from . import homekit_build
from .homekit_nodes import assignment_revision

from .media_sessions import (
    MediaResourceClass, MediaSessionKind, MediaSessionManager, MediaSessionRequest,
)

LOGGER = logging.getLogger(__name__)
BUILD_RECHECK_SECONDS = 10.0
# Local startup can run three 15-second qualification commands per backend
# (QSV and software), then a 15-second source probe before opening control.sock.
LOCAL_STARTUP_GRACE_SECONDS = 120


def camera_port(camera) -> int:
    return camera.homekit.port or 52000 + int.from_bytes(hashlib.sha256(camera.id.encode()).digest()[:4], "big") % 10000


def worker_settings(config, camera, remote: dict | None = None) -> dict:
    """Local workers receive the source URL. Remote workers receive only the feed path."""
    settings = {"version": 1, "bind": config.homekit.bind, "port": camera_port(camera), "name": camera.name,
        "ffmpeg": config.ffmpeg_path, "audio": bool(camera.homekit.audio),
        "allowSoftwareEncoding": bool(camera.homekit.allow_software_encoding),
        "transport": config.capture_rtsp_transport, "remote": remote is not None}
    if remote is None:
        settings["url"] = camera.source_url(camera.homekit.source)
    else:
        settings["feeds"] = remote["feeds"]
        settings["profile"] = remote["profile"]
        settings["encoding"] = remote["encoding"]
    return settings


_GATE_ERRORS = {"offline": "encoder_offline", "software_not_allowed": "encoder_software_not_allowed",
    "waiting": "encoder_waiting", "unavailable": "encoding_unavailable"}


def private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("unsafe HomeKit state directory")


def control(path: Path, method: str, params: dict | None = None) -> dict:
    request = {"version": 1, "id": "supervisor", "method": method, "params": params or {}}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(str(path / "control.sock"))
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = bytearray()
        while b"\n" not in response:
            chunk = connection.recv(8192)
            if not chunk or len(response) + len(chunk) > 32768:
                raise RuntimeError("invalid HomeKit response")
            response.extend(chunk)
    value = json.loads(response.split(b"\n", 1)[0])
    if value.get("version") != 1 or value.get("error") or not isinstance(value.get("result"), dict):
        raise RuntimeError("HomeKit request failed")
    return value["result"]


class CameraWorker:
    def __init__(self, service: HomeKitService, config, camera) -> None:
        self.service, self.config, self.camera = service, config, camera
        self.path = service.directory / camera.id
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"homekit-{camera.id}", daemon=True)
        self.lock = threading.Lock()
        self.state: dict = {"state": "starting", "paired": False, "port": camera_port(camera)}
        self.process: subprocess.Popen | None = None
        self.leases: dict = {}
        self.pending_snapshot: tuple | None = None

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _state(self, **values) -> None:
        with self.lock:
            self.state.update(values)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("HomeKit worker shutdown incomplete")
        if self.process is not None:
            self._terminate()

    def _admission(self, message: dict) -> None:
        token = message.get("token")
        if message.get("version") != 1 or not isinstance(token, str) or len(token) > 40:
            raise RuntimeError("invalid HomeKit media protocol")
        ok = False
        if message.get("method") == "release":
            lease = self.leases.pop(token, None)
            if lease:
                lease.close()
            ok = True
        elif message.get("method") == "started":
            pid = message.get("pid")
            lease = self.leases.get(token)
            if lease is not None and isinstance(pid, int):
                try:
                    if os.getpgid(pid) == self.process.pid:
                        lease.attach_process(pid)
                        ok = True
                except ProcessLookupError:
                    pass
        elif message.get("method") == "snapshot":
            if (self.pending_snapshot is None and not self.stopping.is_set()
                    and self.camera.homekit.encoder != "local"):
                destination = self.path / "remote-snapshot.jpg"
                waiter = self.service.nodes.begin_snapshot(self.camera.id, destination)
                self.pending_snapshot = (message, waiter, time.monotonic() + 12)
                return
        elif message.get("method") == "acquire" and not self.stopping.is_set() and token not in self.leases:
            try:
                purpose = message.get("purpose", "encoder")
                resource = {"encoder": MediaResourceClass.TRANSCODE_PROCESS,
                            "relay": MediaResourceClass.RELAY,
                            "snapshot": MediaResourceClass.JPEG_ENCODER}.get(purpose)
                if resource is None:
                    raise RuntimeError("invalid media purpose")
                lease = self.service.media.acquire(MediaSessionRequest(
                    kind=MediaSessionKind.HOMEKIT, camera_id=self.camera.id,
                    source=self.camera.homekit.source,
                    resources={resource: 1}, metadata={"purpose": purpose},
                    owner_generation=self.service.generation,
                ), blocking=False)
                lease.set_phase("homekit_media")
                self.leases[token] = lease
                self._state(mediaCapacity="available")
                ok = True
            except RuntimeError:
                self._state(mediaCapacity="unavailable")
        self._reply(message, ok)

    def _reply(self, message: dict, ok: bool) -> None:
        self.process.stdin.write(json.dumps({"version": 1, "id": message.get("id"), "ok": ok}).encode() + b"\n")
        self.process.stdin.flush()

    def _poll_snapshot(self, cancel: bool = False) -> None:
        if self.pending_snapshot is None:
            return
        message, waiter, deadline = self.pending_snapshot
        if cancel or waiter[0].is_set() or time.monotonic() >= deadline:
            self.pending_snapshot = None
            self.service.nodes.end_snapshot(self.camera.id, waiter)
            if cancel:
                waiter[1].unlink(missing_ok=True)
            self._reply(message, not cancel and waiter[0].is_set() and waiter[1].is_file())

    def _terminate(self) -> None:
        if self.pending_snapshot is not None:
            _, waiter, _ = self.pending_snapshot
            self.service.nodes.end_snapshot(self.camera.id, waiter)
            self.pending_snapshot = None
        process = self.process
        if process is None:
            return
        # Signal the entire owned session so a crashed Node process cannot leave
        # FFmpeg alive. Reservations are released only after the group is killed.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        (self.path / "remote-snapshot.jpg").unlink(missing_ok=True)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()
        for lease in self.leases.values():
            lease.close("cancelled", "homekit_worker_stopped")
        self.leases.clear()
        self.process = None

    def _run(self) -> None:
        owner = None
        try:
            private_directory(self.service.directory)
            private_directory(self.path)
            # The lock is never deleted. flock releases ownership on crashes.
            fd = os.open(self.path / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            owner = os.fdopen(fd, "w")
            while True:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self._state(state="waiting", error="state_in_use")
                    if self.stopping.wait(1):
                        return
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.bind((self.config.homekit.bind, 0))
            except OSError:
                self._state(state="blocked", error="bind_address_unavailable")
                return
            version = subprocess.run([self.config.homekit.node_path, "--version"], capture_output=True, timeout=5, check=True)
            if not version.stdout.startswith(b"v24."):
                self._state(state="blocked", error="node_24_required")
                return
            encoders = subprocess.run([self.config.ffmpeg_path, "-hide_banner", "-encoders"], capture_output=True, timeout=5, check=True)
            if any(codec not in encoders.stdout for codec in (b"libopus", b"aac", b"mjpeg")):
                self._state(state="blocked", error="encoder_unavailable")
                return
            failures = 0
            while not self.stopping.is_set():
                if not self.service.camera_active(self.camera.id):
                    self._state(state="paused", error="camera_off")
                    self.stopping.wait(1)
                    continue
                revision = assignment_revision(self.config, self.camera)
                decision = self.service.nodes.decision(self.camera, revision)
                if decision not in {"local", "remote"}:
                    self._state(state="blocked" if decision != "waiting" else "starting", error=_GATE_ERRORS[decision])
                    self.stopping.wait(2)
                    continue
                remote = None if decision == "local" else self.service.nodes.remote_for(self.camera, revision)
                if decision == "remote" and remote is None:
                    self._state(state="starting", error="encoder_waiting")
                    self.stopping.wait(2)
                    continue
                release, build_problem = homekit_build.verified_release()
                if release is None:
                    self._state(state="blocked", error=build_problem)
                    self.service.ensure_worker_build()
                    self.stopping.wait(BUILD_RECHECK_SECONDS)
                    continue
                node_left = False
                try:
                    # Only this lock owner may remove a dead worker's stale socket.
                    (self.path / "control.sock").unlink(missing_ok=True)
                    for endpoint in self.path.glob("source-*.sock"):
                        if stat.S_ISSOCK(endpoint.lstat().st_mode):
                            endpoint.unlink()
                    env = {**os.environ, "DEBUG": "", "NODE_DEBUG": ""}
                    self.process = subprocess.Popen([self.config.homekit.node_path, str(release / "dist/src/camera.js")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        start_new_session=True, env=env, umask=0o077, pass_fds=(owner.fileno(),))
                    settings = worker_settings(self.config, self.camera, remote)
                    settings.update(state=str(self.path.resolve()), ownerFd=owner.fileno())
                    self.process.stdin.write(json.dumps(settings).encode() + b"\n")
                    self.process.stdin.flush()
                    self._state(state="starting", error=None)
                    pending = b""
                    control_failures = 0
                    started_at = time.monotonic()
                    # Forgive failed control polls during local qualification.
                    # A successful exchange ends the grace period immediately.
                    startup_grace_until = started_at + LOCAL_STARTUP_GRACE_SECONDS if remote is None else None
                    poll_at = started_at + 1
                    while not self.stopping.is_set() and self.process.poll() is None and self.service.camera_active(self.camera.id):
                        self._poll_snapshot()
                        if any(lease.cancelled() for lease in self.leases.values()):
                            break
                        if select.select([self.process.stdout], [], [], 0.2)[0]:
                            chunk = os.read(self.process.stdout.fileno(), 16384)
                            if not chunk:
                                break
                            pending += chunk
                            if len(pending) > 32768:
                                raise RuntimeError("worker message too large")
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                # HAP may print a startup message; never log it or
                                # mistake it for the versioned admission protocol.
                                if line.startswith(b'{"version":'):
                                    self._admission(json.loads(line))
                        if time.monotonic() >= poll_at:
                            poll_at = time.monotonic() + 1
                            try:
                                snapshot = control(self.path, "status")
                                if remote is not None:
                                    encoder = self.service.nodes.encoder_status(self.camera)
                                    if encoder:
                                        snapshot = {**snapshot, "encoder": encoder}
                                    if self.service.nodes.decision(self.camera, revision) != "remote":
                                        node_left = True
                                        break
                                self._state(state="running", error=None, runtime=snapshot, paired=bool(snapshot.get("paired")))
                                if not snapshot.get("homeKitActive"):
                                    self.service.clear_motion(self.camera.id)
                                    self._poll_snapshot(cancel=True)
                                control(self.path, "motion", {"active": self.service.motion_active(self.camera.id)})
                                failures = 0
                                control_failures = 0
                                startup_grace_until = None
                            except (OSError, ValueError, RuntimeError):
                                self._state(state="starting", error="control_unavailable")
                                control_failures += 1
                                if startup_grace_until is not None:
                                    if time.monotonic() >= startup_grace_until:
                                        break
                                elif control_failures >= 3:
                                    break
                    self.service.clear_motion(self.camera.id)
                    if self.process.poll() == 42:
                        self._state(state="blocked", error="encoding_unavailable")
                        return
                    if node_left:
                        pass
                    elif not self.stopping.is_set():
                        self._state(state="restarting", error="worker_stopped")
                except (OSError, ValueError, RuntimeError):
                    self._state(state="restarting", error="worker_start_failed")
                finally:
                    self._terminate()
                if node_left:
                    continue
                failures += 1
                self.stopping.wait(min(30, 2 ** min(failures, 5)))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            self._state(state="blocked", error="runtime_or_state_unavailable")
            LOGGER.warning("HomeKit camera %s could not start; inspect HomeKit admin status", self.camera.id)
        finally:
            self._terminate()
            if owner:
                owner.close()


class HomeKitService:
    def __init__(self, config, media: MediaSessionManager, generation: str, camera_active: Callable[[str], bool],
                 detection_active: Callable[[str], bool] | None = None) -> None:
        self.config, self.media, self.generation, self.camera_active = config, media, generation, camera_active
        self.directory = Path(config.database_dir or config.storage_dir).resolve() / "homekit"
        self.lock = threading.RLock()
        self.workers: dict[str, CameraWorker] = {}
        self.motion: dict[str, float] = {}
        self.started = False
        self.operations = threading.RLock()
        self.detection_active = detection_active or camera_active
        from .homekit_nodes import NodeRegistry
        self.nodes = NodeRegistry(self.directory)
        self.builder = homekit_build.AutoBuilder()

    def ensure_worker_build(self, *, force: bool = False) -> bool:
        with self.lock:
            config = self.config
            started = self.started
        with self.nodes.lock:
            enrolled = bool(self.nodes.nodes)
        if not force and not (started and (config.homekit.enabled or enrolled)):
            return False
        return self.builder.ensure(config.homekit.node_path, force=force)

    def start(self) -> None:
        with self.operations, self.lock:
            if self.started:
                return
            self.nodes.reopen()
            self.started = True
            self._start_workers()
        self.ensure_worker_build()

    def _start_workers(self) -> None:
        if not self.config.homekit.enabled:
            return
        for camera in self.config.cameras:
            if camera.homekit.enabled:
                worker = CameraWorker(self, self.config, camera)
                self.workers[camera.id] = worker
                worker.start()

    def reconfigure(self, config) -> None:
        with self.operations:
            self._reconfigure(config)
        self.ensure_worker_build()

    def _reconfigure(self, config) -> None:
        self.nodes.reopen()
        def signature(app, camera):
            settings = camera.homekit.model_dump(exclude={"object_motion"})
            return (app.homekit, app.ffmpeg_path, app.capture_rtsp_transport,
                    camera.name, camera.source_url(camera.homekit.source), settings)
        with self.lock:
            previous = self.config
            incoming = {camera.id: camera for camera in config.cameras}
            retiring = [worker for key, worker in self.workers.items()
                if not config.homekit.enabled or key not in incoming
                or signature(previous, worker.camera) != signature(config, incoming[key])]
        # Close outside the service lock: workers consult motion while exiting.
        for worker in retiring:
            worker.close()
        with self.lock:
            for worker in retiring:
                self.workers.pop(worker.camera.id, None)
            self.config = config
            self.motion = {key: value for key, value in self.motion.items()
                if key in incoming and incoming[key].homekit.enabled and incoming[key].homekit.object_motion}
            self.started = True
            if config.homekit.enabled:
                for camera in config.cameras:
                    if camera.homekit.enabled and camera.id not in self.workers:
                        worker = CameraWorker(self, config, camera)
                        self.workers[camera.id] = worker
                        worker.start()

    def close(self) -> None:
        with self.operations:
            self._close()

    def _close(self) -> None:
        with self.lock:
            workers = list(self.workers.values())
        for worker in workers:
            worker.close()
        with self.lock:
            self.workers.clear()
            self.motion.clear()
            self.started = False
            self.nodes.close()

    def observe_objects(self, payload: dict) -> None:
        if not self.config.homekit.enabled:
            return
        # Do not replay stored detections, manual reviews, or stale source events.
        if payload.get("historical") or str(payload.get("source") or "").startswith("manual"):
            return
        if not isinstance(payload.get("timestamp"), str):
            return
        try:
            observed = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
            if observed.tzinfo is None:
                return
            age = datetime.now(timezone.utc).timestamp() - observed.timestamp()
        except (ValueError, KeyError, TypeError):
            return
        objects = payload.get("incident_objects")
        if objects is None:
            objects = payload.get("objects") or []
        if not isinstance(objects, list) or not 0 <= age <= 30 or not any(isinstance(item, dict) and item.get("label")
                and item.get("incident_eligible") is not False for item in objects):
            return
        camera_id = str(payload.get("camera_id") or "")
        if not self.camera_active(camera_id) or not self.detection_active(camera_id):
            return
        with self.lock:
            camera = next((camera for camera in self.config.cameras if camera.id == camera_id), None)
            if camera and camera.homekit.enabled and camera.homekit.object_motion and self.config.homekit.enabled:
                self.motion[camera_id] = time.monotonic() + 10

    def clear_motion(self, camera_id: str) -> None:
        with self.lock:
            self.motion.pop(camera_id, None)

    def motion_active(self, camera_id: str) -> bool:
        if not self.camera_active(camera_id) or not self.detection_active(camera_id):
            self.clear_motion(camera_id)
            return False
        with self.lock:
            return self.motion.get(camera_id, 0) > time.monotonic()

    def node_assignments(self) -> list[dict]:
        with self.lock:
            config = self.config
        if not config.homekit.enabled:
            return []
        assignments = []
        for camera in config.cameras:
            encoder = camera.homekit.encoder or "local"
            if not camera.homekit.enabled or encoder == "local" or not self.camera_active(camera.id):
                continue
            with self.nodes.lock:
                node = self.nodes.nodes.get(encoder)
                hardware = node.get("hardware") if node else ""
            if hardware == "qsv" or (hardware == "software" and camera.homekit.allow_software_encoding):
                assignments.append({"id": camera.id, "encoder": encoder,
                    "url": camera.source_url(camera.homekit.source), "audio": bool(camera.homekit.audio),
                    "transport": config.capture_rtsp_transport,
                    "allowSoftwareEncoding": bool(camera.homekit.allow_software_encoding),
                    "revision": assignment_revision(config, camera)})
        return assignments

    def status(self) -> dict:
        with self.lock:
            config = self.config
            status = {"enabled": config.homekit.enabled, "experimental": True,
                "nodeError": self.nodes.error,
                "cameras": [
                {"id": camera.id, "name": camera.name, "enabled": camera.homekit.enabled,
                 "encoder": camera.homekit.encoder,
                 **(self.workers[camera.id].status() if camera.id in self.workers else {"state": "disabled", "paired": None, "port": camera_port(camera)})}
                for camera in config.cameras
            ]}
        status["nodes"] = self.nodes.public_status(config.cameras)
        status["workerBuild"] = homekit_build.verify()
        status["workerBuildJob"] = self.builder.status()
        return status

    def pairing(self, camera_id: str) -> dict:
        with self.lock:
            worker = self.workers.get(camera_id)
        if worker is None or worker.status().get("state") != "running":
            raise RuntimeError("Enable this camera and wait for HomeKit to start")
        return control(worker.path, "pairing")

    def reset_pairing(self, camera_id: str) -> None:
        with self.operations:
            self._reset_pairing(camera_id)

    def _reset_pairing(self, camera_id: str) -> None:
        with self.lock:
            worker = self.workers.get(camera_id)
        if worker is None:
            raise RuntimeError("Enable this camera before resetting its pairing")
        worker.close()
        try:
            private_directory(worker.path)
            fd = os.open(worker.path / "owner.lock", os.O_RDWR | os.O_NOFOLLOW)
            with os.fdopen(fd, "r+") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                hap = worker.path / "hap"
                if hap.exists():
                    private_directory(hap)
                    shutil.rmtree(hap)
                (worker.path / "identity.json").unlink(missing_ok=True)
        finally:
            # A rejected reset must still restore supervision of the old identity.
            with self.lock:
                camera = next(camera for camera in self.config.cameras if camera.id == camera_id)
                replacement = CameraWorker(self, self.config, camera)
                self.workers[camera_id] = replacement
                self.motion.pop(camera_id, None)
                replacement.start()
