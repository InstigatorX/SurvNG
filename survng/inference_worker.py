"""Self-registering remote inference worker for Proxmox appliances."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid

import numpy as np

from .app.config import DetectorConfig
from .app.security import redact_secret_text
from .app.inference import InferenceSupervisor, InferenceWorkload
from .app.inference_runtime.protocol import (
    INFERENCE_PROTOCOL_VERSION,
    WorkerRegistration,
    decode_frame,
    decode_packet,
    encode_packet,
)
from .app.inference_runtime.model_sync import (
    ModelBundleManifest,
    WorkerModelCache,
)


LOGGER = logging.getLogger("survng.inference-worker")
DEFAULT_WORKER_ID_PATH = Path("/var/lib/survng-inference/worker-id")
DEFAULT_MODEL_CACHE_PATH = Path("/var/lib/survng-inference/models")


def load_or_create_worker_id(path: Path) -> str:
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    if existing:
        WorkerRegistration(
            worker_id=existing,
            roles=["object"],
        )
        return existing
    worker_id = f"worker-{uuid.uuid4().hex}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{worker_id}\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return worker_id


def worker_websocket_url(server_url: str) -> str:
    parsed = urlsplit(server_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError("worker server URL must be HTTP(S) or WS(S)")
    scheme = {
        "http": "ws",
        "https": "wss",
    }.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/inference/workers/connect"):
        path = f"{path}/api/inference/workers/connect"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


_WELCOME_TIMEOUT_SECONDS = 300.0


def _receive_welcome(websocket: Any) -> dict[str, Any]:
    """Wait through model hashing without treating a quiet socket as a failure."""
    deadline = time.monotonic() + _WELCOME_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "timed out waiting for the primary server welcome"
            )
        incoming = websocket.recv(timeout=remaining)
        if not isinstance(incoming, str):
            raise RuntimeError("server returned an invalid worker welcome")
        message = json.loads(incoming)
        if not isinstance(message, dict):
            raise RuntimeError("server returned an invalid worker welcome")
        if message.get("type") == "preparing":
            deadline = time.monotonic() + _WELCOME_TIMEOUT_SECONDS
            continue
        if message.get("type") == "error":
            raise RuntimeError(
                str(message.get("error") or "worker registration was rejected")
            )
        return message


def _role_statuses(
    supervisor: InferenceSupervisor,
    roles: list[str],
) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    if "object" in roles:
        statuses["object"] = supervisor.status()
    if "face" in roles:
        statuses["face"] = supervisor.face_status()
    if "reid" in roles:
        statuses["reid"] = supervisor.reid_status()
    if "depth" in roles:
        statuses["depth"] = supervisor.depth_status()
    return statuses


def _dispatch_request(
    supervisor: InferenceSupervisor,
    message: dict[str, Any],
    frame: np.ndarray | None,
) -> Any:
    role = str(message.get("role") or "")
    operation = str(message.get("operation") or "")
    payload = message.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("inference payload must be an object")
    workload = InferenceWorkload(int(message.get("workload")))
    if role == "object" and operation == "detect":
        if frame is None:
            raise ValueError("object detection requires a frame")
        threshold = payload.get("confidence_threshold")
        methods = {
            InferenceWorkload.INCIDENT_INITIAL: supervisor.detect_initial,
            InferenceWorkload.INCIDENT_REFINEMENT: supervisor.detect_refinement,
            InferenceWorkload.INTERACTIVE: supervisor.detect_interactive,
            InferenceWorkload.TRACKING: supervisor.detect_tracking,
            InferenceWorkload.ENRICHMENT: supervisor.detect_enrichment,
            InferenceWorkload.OFFLINE: supervisor.detect_offline,
        }
        return methods[workload](frame, threshold)
    if role == "face" and operation == "embed":
        if frame is None:
            raise ValueError("face embedding requires a frame")
        return supervisor.embed(frame).astype(np.float32).tolist()
    if role == "face" and operation == "detect_faces":
        if frame is None:
            raise ValueError("face detection requires a frame")
        return supervisor.detect_faces(frame)
    if role == "reid" and operation == "embed_person":
        if frame is None:
            raise ValueError("person ReID requires a frame")
        return supervisor.embed_person(frame).astype(np.float32).tolist()
    if role == "reid" and operation == "embed_reid":
        if frame is None:
            raise ValueError("object ReID requires a frame")
        return supervisor.embed_reid(
            str(payload.get("label") or ""),
            frame,
        ).astype(np.float32).tolist()
    if role == "depth" and operation == "estimate_depth":
        if frame is None:
            raise ValueError("depth estimation requires a frame")
        objects, metadata = supervisor.estimate_depth_for_objects(
            frame,
            list(payload.get("objects") or []),
            frame_offset_s=payload.get("frame_offset_s"),
            include_heatmap=bool(payload.get("include_heatmap", False)),
            workload=workload,
        )
        return {"objects": objects, "metadata": metadata}
    raise ValueError(f"unsupported {role} inference operation: {operation}")


class InferenceWorkerClient:
    def __init__(
        self,
        *,
        server_url: str,
        token: str,
        worker_id: str,
        name: str,
        roles: list[str],
        model_cache_dir: Path = DEFAULT_MODEL_CACHE_PATH,
    ) -> None:
        self.server_url = worker_websocket_url(server_url)
        self.token = token
        self.worker_id = worker_id
        self.name = name
        self.roles = roles
        self.model_cache = WorkerModelCache(model_cache_dir)

    def run_once(self) -> None:
        from websockets.sync.client import connect

        registration = WorkerRegistration(
            worker_id=self.worker_id,
            name=self.name,
            roles=self.roles,
            software_version=os.environ.get("SURVNG_GIT_SHA", ""),
            cached_model_digests=self.model_cache.available_digests(),
        )
        with connect(
            self.server_url,
            additional_headers={
                "Authorization": f"Bearer {self.token}",
            },
            max_size=None,
            open_timeout=10.0,
            close_timeout=5.0,
        ) as websocket:
            websocket.send(registration.model_dump_json())
            welcome = _receive_welcome(websocket)
            if (
                welcome.get("type") != "welcome"
                or int(welcome.get("protocol_version") or 0)
                != INFERENCE_PROTOCOL_VERSION
            ):
                raise RuntimeError("server rejected the worker protocol")
            connection_generation = int(
                welcome["connection_generation"]
            )
            config_generation = str(welcome["config_generation"])
            config = DetectorConfig.model_validate(
                welcome["detector_config"]
            )
            manifest = ModelBundleManifest.model_validate(
                welcome["model_manifest"]
            )
            self.model_cache.receive(websocket, manifest)
            config = self.model_cache.materialize(config, manifest)
            LOGGER.info(
                "Installed %d model file%s (%d bytes) under %s",
                len(manifest.files),
                "" if len(manifest.files) == 1 else "s",
                manifest.total_bytes,
                self.model_cache.bundle_dir / manifest.generation,
            )
            supervisor = InferenceSupervisor(
                config,
                enabled_roles=self.roles,
            )
            try:
                if not supervisor.start():
                    raise RuntimeError("inference engines failed to start")
                websocket.send(json.dumps({
                    "type": "ready",
                    "worker_id": self.worker_id,
                    "connection_generation": connection_generation,
                    "config_generation": config_generation,
                    "statuses": _role_statuses(
                        supervisor,
                        self.roles,
                    ),
                }))
                heartbeat_seconds = max(
                    1.0,
                    float(welcome.get("heartbeat_seconds") or 5.0),
                )
                while True:
                    try:
                        incoming = websocket.recv(
                            timeout=heartbeat_seconds
                        )
                    except TimeoutError:
                        websocket.send(json.dumps({
                            "type": "heartbeat",
                            "worker_id": self.worker_id,
                            "connection_generation": connection_generation,
                            "pending_requests": 0,
                        }))
                        continue
                    if isinstance(incoming, str):
                        control = json.loads(incoming)
                        if (
                            control.get("type") == "heartbeat_ack"
                            and (
                                control.get("accepted") is False
                                or control.get("config_generation")
                                != config_generation
                            )
                        ):
                            return
                        continue
                    if not isinstance(incoming, bytes):
                        continue
                    response = self.handle_packet(
                        supervisor,
                        incoming,
                        connection_generation=connection_generation,
                        config_generation=config_generation,
                    )
                    websocket.send(response)
                    websocket.send(json.dumps({
                        "type": "heartbeat",
                        "worker_id": self.worker_id,
                        "connection_generation": connection_generation,
                        "pending_requests": 0,
                    }))
            finally:
                supervisor.stop()
                supervisor.stop_resource_tracker()

    def handle_packet(
        self,
        supervisor: InferenceSupervisor,
        packet: bytes,
        *,
        connection_generation: int,
        config_generation: str,
    ) -> bytes:
        message, frame_bytes = decode_packet(packet)
        request_id = str(message.get("request_id") or "")
        response: dict[str, Any] = {
            "type": "response",
            "request_id": request_id,
            "connection_generation": connection_generation,
            "ok": False,
        }
        try:
            if message.get("type") != "request" or not request_id:
                raise ValueError("invalid inference request")
            if (
                int(message.get("connection_generation") or 0)
                != connection_generation
            ):
                raise ValueError("stale worker connection generation")
            if message.get("config_generation") != config_generation:
                raise ValueError("worker configuration generation mismatch")
            deadline_ms = int(message.get("deadline_unix_ms") or 0)
            if deadline_ms <= int(time.time() * 1000):
                raise TimeoutError("inference request deadline expired")
            frame = decode_frame(message, frame_bytes)
            started = time.perf_counter()
            response["result"] = _dispatch_request(
                supervisor,
                message,
                frame,
            )
            response["inference_ms"] = round(
                (time.perf_counter() - started) * 1000,
                1,
            )
            response["ok"] = True
        except Exception as error:
            response["error"] = (
                "remote inference request failed: "
                f"{type(error).__name__}"
            )
        return encode_packet(response)

    def run_forever(self) -> None:
        delay = 1.0
        while True:
            try:
                self.run_once()
                delay = 1.0
            except KeyboardInterrupt:
                raise
            except Exception as error:
                LOGGER.warning(
                    "Inference worker disconnected (%s: %s); retrying in %.1fs",
                    type(error).__name__,
                    redact_secret_text(error)[:300],
                    delay,
                )
                time.sleep(delay)
                delay = min(30.0, delay * 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SurvNG remote inference worker",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("SURVNG_INFERENCE_SERVER", ""),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SURVNG_INFERENCE_TOKEN", ""),
    )
    parser.add_argument(
        "--worker-id-file",
        type=Path,
        default=Path(
            os.environ.get(
                "SURVNG_INFERENCE_WORKER_ID_FILE",
                str(DEFAULT_WORKER_ID_PATH),
            )
        ),
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "SURVNG_INFERENCE_MODEL_CACHE_DIR",
                str(DEFAULT_MODEL_CACHE_PATH),
            )
        ),
    )
    parser.add_argument(
        "--name",
        default=os.environ.get(
            "SURVNG_INFERENCE_WORKER_NAME",
            socket.gethostname(),
        ),
    )
    parser.add_argument(
        "--roles",
        default=os.environ.get(
            "SURVNG_INFERENCE_WORKER_ROLES",
            "object,face,reid,depth",
        ),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.server or not args.token:
        parser.error("--server and --token are required")
    roles = [
        item.strip().lower()
        for item in args.roles.split(",")
        if item.strip()
    ]
    registration = WorkerRegistration(
        worker_id="validation",
        roles=roles,
    )
    client = InferenceWorkerClient(
        server_url=args.server,
        token=args.token,
        worker_id=load_or_create_worker_id(args.worker_id_file),
        name=args.name,
        roles=list(registration.roles),
        model_cache_dir=args.model_cache_dir,
    )
    logging.basicConfig(
        level=os.environ.get("SURVNG_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.once:
        client.run_once()
    else:
        client.run_forever()


if __name__ == "__main__":
    main()
