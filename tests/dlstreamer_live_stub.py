"""Emit one framed live-capture sample for unit tests. URL is read from stdin."""

from __future__ import annotations

import sys
import time

from survng.app.dlstreamer_protocol import (
    TYPE_DETECTIONS,
    TYPE_STATUS,
    encode_frame,
    encode_json,
)


def main() -> int:
    sys.stdin.readline()
    pixels = bytes((20, 40, 200)) * 4
    stdout = sys.stdout.buffer
    stdout.write(
        encode_json(
            TYPE_STATUS,
            {
                "ok": True,
                "decoder_elements": ["avdec_h264"],
                "source_element": "uridecodebin3",
            },
        )
    )
    stdout.write(
        encode_frame(width=2, height=2, sequence=1, pts=0.05, pixels=pixels)
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
        )
    )
    stdout.flush()
    time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
