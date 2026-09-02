from __future__ import annotations

import pytest

from survng.app.dlstreamer_protocol import (
    MAX_MESSAGE_BYTES,
    TYPE_DETECTIONS,
    TYPE_FRAME,
    TYPE_JPEG,
    TYPE_STATUS,
    MessageReader,
    ProtocolError,
    decode_frame_payload,
    decode_jpeg_payload,
    decode_json_payload,
    encode_frame,
    encode_jpeg,
    encode_json,
)


def test_frame_round_trip_preserves_bgr_pixels() -> None:
    pixels = bytes((20, 40, 200))
    encoded = encode_frame(width=1, height=1, sequence=7, pts=1.25, pixels=pixels)
    reader = MessageReader()
    reader.feed(encoded)
    message_type, payload = reader.pop() or (0, b"")

    assert message_type == TYPE_FRAME
    width, height, sequence, pts, decoded = decode_frame_payload(payload)
    assert (width, height, sequence, pts, decoded) == (1, 1, 7, 1.25, pixels)
    assert reader.pop() is None


def test_frame_round_trip_preserves_gray_pixels() -> None:
    pixels = bytes((9, 18, 27, 36))
    encoded = encode_frame(width=2, height=2, sequence=3, pts=0.5, pixels=pixels)
    reader = MessageReader()
    reader.feed(encoded)
    message_type, payload = reader.pop() or (0, b"")

    assert message_type == TYPE_FRAME
    width, height, sequence, pts, decoded = decode_frame_payload(payload)
    assert (width, height, sequence, pts, decoded) == (2, 2, 3, 0.5, pixels)


def test_jpeg_round_trip_preserves_bytes() -> None:
    jpeg = b"\xff\xd8\xff\xd9payload"
    encoded = encode_jpeg(width=8, height=4, sequence=2, pts=0.25, jpeg=jpeg)
    reader = MessageReader()
    reader.feed(encoded)
    message_type, payload = reader.pop() or (0, b"")

    assert message_type == TYPE_JPEG
    width, height, sequence, pts, decoded = decode_jpeg_payload(payload)
    assert (width, height, sequence, pts, decoded) == (8, 4, 2, 0.25, jpeg)


def test_json_messages_round_trip() -> None:
    encoded = encode_json(
        TYPE_DETECTIONS,
        {"objects": [{"label": "person", "confidence": 0.9}]},
    )
    reader = MessageReader()
    reader.feed(encoded[:4])
    assert reader.pop() is None
    reader.feed(encoded[4:])
    message_type, payload = reader.pop() or (0, b"")

    assert message_type == TYPE_DETECTIONS
    assert decode_json_payload(payload)["objects"][0]["label"] == "person"


def test_status_error_payload_is_json() -> None:
    encoded = encode_json(TYPE_STATUS, {"ok": False, "error": "timed out"})
    reader = MessageReader()
    reader.feed(encoded)
    message_type, payload = reader.pop() or (0, b"")

    assert message_type == TYPE_STATUS
    assert decode_json_payload(payload) == {"error": "timed out", "ok": False}


def test_invalid_magic_is_rejected() -> None:
    reader = MessageReader()
    reader.feed(b"XXXX\x01\x00\x00\x00\x00")
    with pytest.raises(ProtocolError, match="invalid framing"):
        reader.pop()


def test_oversized_message_is_rejected() -> None:
    header = bytearray(9)
    header[:4] = b"NGDS"
    header[4] = TYPE_FRAME
    header[5:9] = (MAX_MESSAGE_BYTES + 1).to_bytes(4, "big")
    reader = MessageReader()
    reader.feed(bytes(header))
    with pytest.raises(ProtocolError, match="256 MiB"):
        reader.pop()
