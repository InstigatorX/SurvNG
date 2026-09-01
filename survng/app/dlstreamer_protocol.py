"""Length-prefixed live-capture messages between the DL Streamer child and parent."""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Iterable


MAGIC = b"NGDS"
MESSAGE_HEADER = struct.Struct("!4sBI")
FRAME_HEADER = struct.Struct("!IIId")
TYPE_FRAME = 1
TYPE_DETECTIONS = 2
TYPE_STATUS = 3
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when the live-capture child emits an invalid message."""


def encode_message(message_type: int, payload: bytes) -> bytes:
    if message_type not in {TYPE_FRAME, TYPE_DETECTIONS, TYPE_STATUS}:
        raise ProtocolError(f"unsupported live-capture message type {message_type}")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("live-capture message exceeded 256 MiB")
    return MESSAGE_HEADER.pack(MAGIC, message_type, len(payload)) + payload


def encode_frame(
    *,
    width: int,
    height: int,
    sequence: int,
    pts: float,
    pixels: bytes,
) -> bytes:
    expected = width * height * 3
    if width <= 0 or height <= 0 or len(pixels) != expected:
        raise ProtocolError("live-capture frame payload does not match BGR dimensions")
    return encode_message(
        TYPE_FRAME,
        FRAME_HEADER.pack(width, height, sequence, float(pts)) + pixels,
    )


def encode_json(message_type: int, payload: dict[str, Any]) -> bytes:
    return encode_message(
        message_type,
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )


def decode_frame_payload(payload: bytes) -> tuple[int, int, int, float, bytes]:
    if len(payload) < FRAME_HEADER.size:
        raise ProtocolError("truncated live-capture frame header")
    width, height, sequence, pts = FRAME_HEADER.unpack_from(payload)
    pixels = payload[FRAME_HEADER.size :]
    expected = width * height * 3
    if width <= 0 or height <= 0 or len(pixels) != expected:
        raise ProtocolError("live-capture frame payload does not match BGR dimensions")
    return width, height, sequence, float(pts), pixels


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
        if message_type not in {TYPE_FRAME, TYPE_DETECTIONS, TYPE_STATUS}:
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
