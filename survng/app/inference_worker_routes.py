from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
import json
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .inference_runtime.protocol import (
    ProtocolError,
    WorkerHeartbeat,
    WorkerReady,
    WorkerRegistration,
    decode_packet,
    encode_binary_packet,
)
from .inference_runtime.model_sync import (
    MODEL_SYNC_CHUNK_BYTES,
    ModelBundleCatalog,
    ModelSyncError,
    PreparedModelBundle,
)
from .inference_runtime.registry import RemoteInferenceRegistry
from .inference_runtime.types import InferenceUnavailable


_REGISTRATION_TIMEOUT_SECONDS = 10.0
_MODEL_SYNC_TIMEOUT_SECONDS = 300.0
_INITIAL_WORKER_LEASE_SECONDS = 600.0


async def _send_model_bundle(
    websocket: WebSocket,
    registration: WorkerRegistration,
    prepared: PreparedModelBundle,
) -> None:
    cached = frozenset(registration.cached_model_digests)
    sent: set[str] = set()
    for model_file in prepared.manifest.files:
        if model_file.digest in cached or model_file.digest in sent:
            continue
        source = prepared.sources[model_file.digest]
        offset = 0
        with source.open("rb") as handle:
            while chunk := await asyncio.to_thread(
                handle.read,
                MODEL_SYNC_CHUNK_BYTES,
            ):
                await websocket.send_bytes(encode_binary_packet(
                    {
                        "type": "model_chunk",
                        "digest": model_file.digest,
                        "offset": offset,
                        "total_size": model_file.size,
                    },
                    chunk,
                ))
                offset += len(chunk)
        if offset != model_file.size:
            raise InferenceUnavailable(
                "model artifact changed during synchronization"
            )
        sent.add(model_file.digest)
    await websocket.send_json({
        "type": "model_sync_complete",
        "generation": prepared.manifest.generation,
    })


class WebSocketRegistryTransport:
    """Thread-safe request bridge to one async worker WebSocket."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._outbound: asyncio.Queue[bytes | str | None] = asyncio.Queue()
        self._lock = threading.RLock()
        self._pending: dict[str, Future[bytes]] = {}
        self._closed_reason = ""

    def request(self, packet: bytes, timeout: float) -> bytes:
        message, _frame = decode_packet(packet)
        request_id = str(message.get("request_id") or "")
        if not request_id:
            raise InferenceUnavailable("remote inference request has no id")
        future: Future[bytes] = Future()
        with self._lock:
            if self._closed_reason:
                raise InferenceUnavailable(self._closed_reason)
            self._pending[request_id] = future
        self._loop.call_soon_threadsafe(self._outbound.put_nowait, packet)
        try:
            return future.result(timeout=max(0.0, timeout))
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def deliver(self, packet: bytes) -> bool:
        try:
            message, _frame = decode_packet(packet)
        except ProtocolError:
            return False
        request_id = str(message.get("request_id") or "")
        if message.get("type") != "response" or not request_id:
            return False
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(packet)
        return True

    async def send_loop(self, websocket: WebSocket) -> None:
        while True:
            packet = await self._outbound.get()
            if packet is None:
                return
            if isinstance(packet, bytes):
                await websocket.send_bytes(packet)
            else:
                await websocket.send_text(packet)

    def send_control(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, separators=(",", ":"))
        self._loop.call_soon_threadsafe(
            self._outbound.put_nowait,
            encoded,
        )

    def close(self, reason: str) -> None:
        with self._lock:
            if self._closed_reason:
                return
            self._closed_reason = reason or "remote worker disconnected"
            pending = list(self._pending.values())
        error = InferenceUnavailable(self._closed_reason)
        for future in pending:
            if not future.done():
                future.set_exception(error)
        self._loop.call_soon_threadsafe(self._outbound.put_nowait, None)


@dataclass(frozen=True, slots=True)
class InferenceWorkerRouteDependencies:
    registry: RemoteInferenceRegistry
    authenticate: Callable[[str], bool]
    model_catalog: ModelBundleCatalog


def create_inference_worker_router(
    deps: InferenceWorkerRouteDependencies,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/api/inference/workers/connect")
    async def connect_inference_worker(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        if not deps.authenticate(authorization):
            await websocket.close(code=1008, reason="invalid worker credential")
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        transport = WebSocketRegistryTransport(loop)
        worker_id = ""
        connection_generation = 0
        sender: asyncio.Task[None] | None = None
        try:
            raw_registration = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=_REGISTRATION_TIMEOUT_SECONDS,
            )
            registration = WorkerRegistration.model_validate_json(
                raw_registration
            )
            worker_id = registration.worker_id
            config_snapshot = deps.registry.config_snapshot()
            prepared = await asyncio.to_thread(
                deps.model_catalog.prepare,
                config_snapshot,
                registration.roles,
            )
            welcome = deps.registry.register(
                registration,
                transport,
                config=config_snapshot,
                config_generation=prepared.config_generation,
                initial_lease_seconds=_INITIAL_WORKER_LEASE_SECONDS,
            )
            welcome["detector_config"] = prepared.config.model_dump(
                mode="json"
            )
            welcome["model_manifest"] = prepared.manifest.model_dump(
                mode="json"
            )
            connection_generation = int(
                welcome["connection_generation"]
            )
            await websocket.send_json(welcome)
            await asyncio.wait_for(
                _send_model_bundle(websocket, registration, prepared),
                timeout=_MODEL_SYNC_TIMEOUT_SECONDS,
            )
            sender = asyncio.create_task(transport.send_loop(websocket))
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break
                text = message.get("text")
                packet = message.get("bytes")
                if packet is not None:
                    if not transport.deliver(packet):
                        raise ProtocolError(
                            "worker sent an unknown inference response"
                        )
                    continue
                if text is None:
                    raise ProtocolError("worker sent an empty message")
                control = json.loads(text)
                if not isinstance(control, dict):
                    raise ProtocolError(
                        "worker control message must be an object"
                    )
                if control.get("type") == "ready":
                    accepted = await asyncio.to_thread(
                        deps.registry.mark_ready,
                        WorkerReady.model_validate(control),
                    )
                    if not accepted:
                        raise InferenceUnavailable(
                            "worker readiness was rejected"
                        )
                elif control.get("type") == "heartbeat":
                    accepted = deps.registry.heartbeat(
                        WorkerHeartbeat.model_validate(control)
                    )
                    config_generation = await asyncio.to_thread(
                        deps.registry.config_generation
                    )
                    transport.send_control({
                        "type": "heartbeat_ack",
                        "accepted": accepted,
                        "config_generation": config_generation,
                    })
                else:
                    raise ProtocolError(
                        "worker sent an unknown control message"
                    )
        except (
            asyncio.TimeoutError,
            json.JSONDecodeError,
            ProtocolError,
            ModelSyncError,
            ValidationError,
            InferenceUnavailable,
        ):
            await websocket.close(code=1008, reason="invalid worker protocol")
        except WebSocketDisconnect:
            pass
        finally:
            if worker_id and connection_generation:
                deps.registry.unregister(
                    worker_id,
                    connection_generation,
                )
            transport.close("remote worker disconnected")
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    return router
