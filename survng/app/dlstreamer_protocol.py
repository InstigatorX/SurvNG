"""Length-prefixed live-capture messages between the DL Streamer child and parent."""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Iterable


MAGIC = b"NGDS"
MESSAGE_HEADER = struct.Struct("!4sBI")
FRAME_HEADER = struct.Struct("!IIId")
STREAM_ID_HEADER = struct.Struct("!H")
TYPE_FRAME = 1
TYPE_DETECTIONS = 2
TYPE_STATUS = 3
TYPE_JPEG = 4
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_STREAM_ID_BYTES = 65535
_MESSAGE_TYPES = {TYPE_FRAME, TYPE_DETECTIONS, TYPE_STATUS, TYPE_JPEG}


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


def decode_stream_payload(payload: bytes) -> tuple[str, bytes]:
    if len(payload) < STREAM_ID_HEADER.size:
        raise ProtocolError("truncated live-capture stream id")
    (length,) = STREAM_ID_HEADER.unpack_from(payload)
    start = STREAM_ID_HEADER.size
    end = start + length
    if length < 1 or end > len(payload):
        raise ProtocolError("truncated live-capture stream id")
    try:
        stream_id = payload[start:end].decode("utf-8")
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
    pixel_count = width * height
    if width <= 0 or height <= 0 or len(pixels) not in {pixel_count, pixel_count * 3}:
        raise ProtocolError("live-capture frame payload does not match gray or BGR dimensions")
    return encode_message(
        TYPE_FRAME,
        _payload(FRAME_HEADER.pack(width, height, sequence, float(pts)) + pixels, stream_id),
    )


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


def decode_frame_payload(payload: bytes) -> tuple[int, int, int, float, bytes]:
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
    """Buffer stdout from the live-capture child into complete messages."""

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
            raise ProtocolError("live-capture child emitted invalid framing")
        if message_type not in _MESSAGE_TYPES:
            raise ProtocolError(f"unsupported live-capture message type {message_type}")
        if length > MAX_MESSAGE_BYTES:
            raise ProtocolError("live-capture message exceeded 256 MiB")
        total = MESSAGE_HEADER.size + length
        if len(self._buffer) < total:
            return None
        payload = bytes(self._buffer[MESSAGE_HEADER.size : total])
        del self._buffer[:total]
        return message_type, payload


def write_messages(stream: BinaryIO, messages: Iterable[bytes]) -> None:
    for message in messages:
        stream.write(message)
    stream.flush()
