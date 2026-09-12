"""Supervisor-mode live-capture stub. JSON add/remove on stdin, prefixed NGDS."""

from __future__ import annotations

import json
import sys
import time

from survng.app.dlstreamer_protocol import (
    TYPE_DETECTIONS,
    TYPE_STATUS,
    encode_frame,
    encode_jpeg,
    encode_json,
)


def _emit(stream_id: str) -> None:
    pixels = bytes((20, 40, 200)) * 4
    stdout = sys.stdout.buffer
    stdout.write(
        encode_json(
            TYPE_STATUS,
            {
                "ok": True,
                "detect": True,
                "decoder_elements": ["avdec_h264"],
                "source_element": "uridecodebin3",
                "hardware_decoder_selected": False,
                "preprocess_backend": "opencv",
                "first_frame_ms": 12.5,
                "qualifier_format": "GRAY8",
                "qualifier_width": 2,
                "jpeg_preview": True,
                "model_instance_id": "survng-yolo-GPU",
                "shared_detect": True,
            },
            stream_id=stream_id,
        )
    )
    stdout.write(
        encode_frame(
            width=2,
            height=2,
            sequence=1,
            pts=0.05,
            pixels=pixels,
            stream_id=stream_id,
        )
    )
    stdout.write(
        encode_jpeg(
            width=2,
            height=2,
            sequence=1,
            pts=0.05,
            jpeg=b"\xff\xd8stub-jpeg\xff\xd9",
            stream_id=stream_id,
        )
    )
    stdout.write(
        encode_json(
            TYPE_DETECTIONS,
            {
                "objects": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "box": {"x1": 0, "y1": 0, "x2": 2, "y2": 2},
                    }
                ]
            },
            stream_id=stream_id,
        )
    )
    stdout.flush()


def main() -> int:
    active: set[str] = set()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        try:
            command = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(command, dict):
            continue
        stream_id = str(command.get("stream_id") or "").strip()
        operation = str(command.get("op") or "").strip()
        if operation == "remove":
            active.discard(stream_id)
            continue
        if operation != "add" or not stream_id:
            continue
        active.add(stream_id)
        _emit(stream_id)
    time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
