from __future__ import annotations

import base64
import json
import math
import struct
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .types import MAX_INFERENCE_FRAME_BYTES


INFERENCE_PROTOCOL_VERSION = 1
MAX_PROTOCOL_HEADER_BYTES = 2 * 1024 * 1024
MAX_PROTOCOL_PACKET_BYTES = (
    4 + MAX_PROTOCOL_HEADER_BYTES + MAX_INFERENCE_FRAME_BYTES
)
_HEADER_LENGTH = struct.Struct("!I")
_BYTES_MARKER = "__survng_bytes_b64__"
WorkerRole = Literal["object", "face", "reid", "depth"]


class ProtocolError(ValueError):
    pass


class WorkerRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["register"] = "register"
    protocol_version: int = INFERENCE_PROTOCOL_VERSION
    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    name: str = Field(default="", max_length=128)
    roles: list[WorkerRole] = Field(min_length=1, max_length=4)
    slots: int = Field(default=1, ge=1, le=64)
    max_frame_bytes: int = Field(
        default=MAX_INFERENCE_FRAME_BYTES,
        ge=1,
        le=MAX_INFERENCE_FRAME_BYTES,
    )
    software_version: str = Field(default="", max_length=128)
    devices: list[str] = Field(default_factory=list, max_length=32)


class WorkerReady(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ready"] = "ready"
    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    connection_generation: int = Field(ge=1)
    config_generation: str = Field(min_length=1, max_length=128)
    statuses: dict[str, Any] = Field(default_factory=dict)


class WorkerHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["heartbeat"] = "heartbeat"
    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    connection_generation: int = Field(ge=1)
    pending_requests: int = Field(default=0, ge=0, le=100000)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {_BYTES_MARKER: base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("protocol values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProtocolError(f"unsupported protocol value: {type(value).__name__}")


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_BYTES_MARKER}:
            encoded = value[_BYTES_MARKER]
            if not isinstance(encoded, str):
                raise ProtocolError("invalid encoded binary value")
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ProtocolError("invalid encoded binary value") from error
        return {str(key): _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


def encode_packet(
    message: dict[str, Any],
    *,
    frame: np.ndarray | None = None,
) -> bytes:
    payload = dict(message)
    frame_bytes = b""
    if frame is not None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or any(int(value) <= 0 for value in frame.shape)
        ):
            raise ProtocolError(
                "inference frame must be a non-empty uint8 BGR image"
            )
        contiguous = np.ascontiguousarray(frame)
        frame_bytes = contiguous.tobytes()
        if len(frame_bytes) > MAX_INFERENCE_FRAME_BYTES:
            raise ProtocolError(
                f"inference frame exceeds {MAX_INFERENCE_FRAME_BYTES} bytes"
            )
        payload["frame"] = {
            "shape": [int(value) for value in contiguous.shape],
            "dtype": "uint8",
            "byte_count": len(frame_bytes),
        }
    try:
        header = json.dumps(
            _json_safe(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError("protocol message is not serializable") from error
    if not header or len(header) > MAX_PROTOCOL_HEADER_BYTES:
        raise ProtocolError("protocol header is empty or too large")
    return _HEADER_LENGTH.pack(len(header)) + header + frame_bytes


def decode_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    if not isinstance(packet, bytes):
        raise ProtocolError("protocol packet must be bytes")
    if len(packet) < _HEADER_LENGTH.size:
        raise ProtocolError("protocol packet is truncated")
    if len(packet) > MAX_PROTOCOL_PACKET_BYTES:
        raise ProtocolError("protocol packet is too large")
    (header_length,) = _HEADER_LENGTH.unpack_from(packet)
    if header_length <= 0 or header_length > MAX_PROTOCOL_HEADER_BYTES:
        raise ProtocolError("invalid protocol header size")
    header_end = _HEADER_LENGTH.size + header_length
    if header_end > len(packet):
        raise ProtocolError("protocol header is truncated")
    try:
        raw = json.loads(packet[_HEADER_LENGTH.size:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid protocol JSON") from error
    if not isinstance(raw, dict):
        raise ProtocolError("protocol message must be an object")
    message = _json_restore(raw)
    frame_bytes = packet[header_end:]
    frame_metadata = message.get("frame")
    if frame_metadata is None:
        if frame_bytes:
            raise ProtocolError("unexpected protocol frame bytes")
        return message, b""
    if not isinstance(frame_metadata, dict):
        raise ProtocolError("invalid frame metadata")
    try:
        shape = tuple(int(value) for value in frame_metadata.get("shape") or ())
        byte_count = int(frame_metadata.get("byte_count") or 0)
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid frame metadata") from error
    if (
        frame_metadata.get("dtype") != "uint8"
        or len(shape) != 3
        or shape[2] != 3
        or any(value <= 0 for value in shape)
        or math.prod(shape) != byte_count
        or byte_count != len(frame_bytes)
        or byte_count > MAX_INFERENCE_FRAME_BYTES
    ):
        raise ProtocolError("invalid inference frame")
    return message, frame_bytes


def decode_frame(message: dict[str, Any], frame_bytes: bytes) -> np.ndarray | None:
    metadata = message.get("frame")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ProtocolError("invalid frame metadata")
    shape = tuple(int(value) for value in metadata["shape"])
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
