"""Length-prefixed live-capture messages between the DL Streamer child and parent."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
from typing import Any, BinaryIO, Iterable


PROTOCOL_FD_ENV = "SURVNG_DLSTREAMER_PROTOCOL_FD"
MAGIC = b"NGDS"
MESSAGE_HEADER = struct.Struct("!4sBI")
FRAME_HEADER = struct.Struct("!IIId")
STREAM_ID_HEADER = struct.Struct("!H")
TYPE_FRAME = 1
TYPE_DETECTIONS = 2
TYPE_STATUS = 3
TYPE_JPEG = 4
# Supervisor-wide startup failure: JSON without a per-stream envelope.
TYPE_FATAL = 5
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_STREAM_ID_BYTES = 65535
_MESSAGE_TYPES = {TYPE_FRAME, TYPE_DETECTIONS, TYPE_STATUS, TYPE_JPEG, TYPE_FATAL}


class ProtocolError(RuntimeError):
    """Raised when the live-capture child emits an invalid message."""


def encode_message(message_type: int, payload: bytes) -> bytes:
    if message_type not in _MESSAGE_TYPES:
        raise ProtocolError(f"unsupported live-capture message type {message_type}")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("live-capture message exceeded 256 MiB")
    return MESSAGE_HEADER.pack(MAGIC, message_type, len(payload)) + payload


def encode_stream_payload(stream_id: str, payload: bytes) -> bytes:
    encoded = stream_id.encode("utf-8")
    if not encoded:
        raise ProtocolError("live-capture stream id is empty")
    if len(encoded) > MAX_STREAM_ID_BYTES:
        raise ProtocolError("live-capture stream id exceeded 65535 bytes")
    return STREAM_ID_HEADER.pack(len(encoded)) + encoded + payload


def decode_stream_payload(payload: bytes | memoryview) -> tuple[str, bytes | memoryview]:
    if len(payload) < STREAM_ID_HEADER.size:
        raise ProtocolError("truncated live-capture stream id")
    (length,) = STREAM_ID_HEADER.unpack_from(payload)
    start = STREAM_ID_HEADER.size
    end = start + length
    if length < 1 or end > len(payload):
        raise ProtocolError("truncated live-capture stream id")
    try:
        stream_id = bytes(payload[start:end]).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("live-capture stream id is invalid") from error
    if not stream_id:
        raise ProtocolError("live-capture stream id is empty")
    return stream_id, payload[end:]


def _payload(payload: bytes, stream_id: str) -> bytes:
    return encode_stream_payload(stream_id, payload) if stream_id else payload


def encode_frame(
    *,
    width: int,
    height: int,
    sequence: int,
    pts: float,
    pixels: bytes,
    stream_id: str = "",
) -> bytes:
    return b"".join(encode_frame_parts(width=width, height=height, sequence=sequence,
                                     pts=pts, pixels=pixels, stream_id=stream_id))


def encode_frame_parts(
    *, width: int, height: int, sequence: int, pts: float,
    pixels: bytes, stream_id: str = "",
) -> tuple[bytes, bytes]:
    """Preserve framing without concatenating the large pixel allocation."""
    pixel_count = width * height
    if width <= 0 or height <= 0 or len(pixels) not in {pixel_count, pixel_count * 3}:
        raise ProtocolError("live-capture frame payload does not match gray or BGR dimensions")
    prefix = _payload(FRAME_HEADER.pack(width, height, sequence, float(pts)), stream_id)
    length = len(prefix) + len(pixels)
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError("live-capture message exceeded 256 MiB")
    return MESSAGE_HEADER.pack(MAGIC, TYPE_FRAME, length) + prefix, pixels


def encode_jpeg(
    *,
    width: int,
    height: int,
    sequence: int,
    pts: float,
    jpeg: bytes,
    stream_id: str = "",
) -> bytes:
    if width <= 0 or height <= 0 or not jpeg:
        raise ProtocolError("live-capture JPEG payload is empty")
    return encode_message(
        TYPE_JPEG,
        _payload(FRAME_HEADER.pack(width, height, sequence, float(pts)) + jpeg, stream_id),
    )


def encode_json(
    message_type: int,
    payload: dict[str, Any],
    *,
    stream_id: str = "",
) -> bytes:
    return encode_message(
        message_type,
        _payload(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            stream_id,
        ),
    )


def encode_detection_snapshot(
    *,
    source_pts: float,
    inference_sequence: int,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
    stream_id: str = "",
) -> bytes:
    """Encode one authoritative detector result, including an empty result."""
    if width <= 0 or height <= 0 or inference_sequence <= 0:
        raise ProtocolError("invalid detection snapshot identity")
    return encode_json(
        TYPE_DETECTIONS,
        {
            "schema_version": 1,
            "source_pts": float(source_pts),
            "inference_sequence": inference_sequence,
            "width": width,
            "height": height,
            "objects": objects,
        },
        stream_id=stream_id,
    )


def decode_frame_payload(payload: bytes | memoryview) -> tuple[int, int, int, float, bytes | memoryview]:
    if len(payload) < FRAME_HEADER.size:
        raise ProtocolError("truncated live-capture frame header")
    width, height, sequence, pts = FRAME_HEADER.unpack_from(payload)
    pixels = payload[FRAME_HEADER.size :]
    pixel_count = width * height
    if width <= 0 or height <= 0 or len(pixels) not in {pixel_count, pixel_count * 3}:
        raise ProtocolError("live-capture frame payload does not match gray or BGR dimensions")
    return width, height, sequence, float(pts), pixels


def decode_jpeg_payload(payload: bytes) -> tuple[int, int, int, float, bytes]:
    if len(payload) < FRAME_HEADER.size + 2:
        raise ProtocolError("truncated live-capture JPEG header")
    width, height, sequence, pts = FRAME_HEADER.unpack_from(payload)
    jpeg = payload[FRAME_HEADER.size :]
    if width <= 0 or height <= 0 or not jpeg:
        raise ProtocolError("live-capture JPEG payload is empty")
    return width, height, sequence, float(pts), jpeg


def decode_json_payload(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("live-capture JSON message is invalid") from error
    if not isinstance(value, dict):
        raise ProtocolError("live-capture JSON message must be an object")
    return value


class MessageReader:
    """Buffer the live-capture data stream into complete messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buffer.extend(chunk)
        if len(self._buffer) > MAX_MESSAGE_BYTES + MESSAGE_HEADER.size:
            raise ProtocolError("live-capture frame exceeded 256 MiB")

    def pop(self) -> tuple[int, bytes] | None:
        if len(self._buffer) < MESSAGE_HEADER.size:
            return None
        magic, message_type, length = MESSAGE_HEADER.unpack_from(self._buffer)
        if magic != MAGIC:
            raise ProtocolError(
                f"live-capture child emitted invalid framing (buffered_bytes={len(self._buffer)})"
            )
        if message_type not in _MESSAGE_TYPES:
            raise ProtocolError(f"unsupported live-capture message type {message_type}")
        if length > MAX_MESSAGE_BYTES:
            raise ProtocolError("live-capture message exceeded 256 MiB")
        total = MESSAGE_HEADER.size + length
        if len(self._buffer) < total:
            return None
        # Copy out once before resizing the mutable input buffer. A bytearray
        # slice followed by bytes() otherwise copies every frame twice.
        payload = bytes(memoryview(self._buffer)[MESSAGE_HEADER.size : total])
        del self._buffer[:total]
        return message_type, payload


def write_messages(stream: BinaryIO, messages: Iterable[bytes]) -> None:
    for message in messages:
        stream.write(message)
    stream.flush()


class ProtocolWriter:
    """Serialize complete messages; a damaged connection is never reused."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self._lock = threading.Lock()
        self._failed = False

    def send(self, message: bytes | tuple[bytes, ...]) -> None:
        with self._lock:
            if self._failed:
                raise BrokenPipeError("live-capture protocol writer failed")
            try:
                parts = message if isinstance(message, tuple) else (message,)
                for part in parts:
                    remaining = memoryview(part)
                    while remaining:
                        written = self.stream.write(remaining)
                        if written is None or written <= 0:
                            raise BrokenPipeError("live-capture protocol write made no progress")
                        remaining = remaining[written:]
                self.stream.flush()
            except BaseException:
                self._failed = True
                raise


def open_protocol_output() -> BinaryIO:
    """Use an inherited data pipe; retain stdout for standalone CLI consumers."""
    descriptor = os.environ.get(PROTOCOL_FD_ENV)
    if descriptor is None:
        return sys.stdout.buffer
    fd = int(descriptor)
    if fd < 3:
        raise ValueError("live-capture protocol descriptor must be separate from stdio")
    # Called after the native-library-path re-exec. Do not leak into later execs.
    os.set_inheritable(fd, False)
    return os.fdopen(fd, "wb", buffering=0)
