from __future__ import annotations

import io
from fractions import Fraction

import pytest

from survng import gstreamer_capture_probe as probe


def test_camera_url_is_read_from_stdin_without_cli_argument() -> None:
    value = probe._read_camera_url(
        "-",
        io.StringIO("rtsp://user:secret@camera/live\n"),
    )

    assert value == "rtsp://user:secret@camera/live"


def test_camera_url_rejects_unsupported_protocol_without_echoing_secret() -> None:
    with pytest.raises(ValueError) as error:
        probe._read_camera_url(
            "-",
            io.StringIO("file://user:secret@camera/live\n"),
        )

    assert "secret" not in str(error.value)


def test_probe_errors_redact_url_passwords() -> None:
    redacted = probe._redact_error(
        "failed to open rtsp://admin:camera-secret@camera/live"
    )

    assert redacted == "failed to open rtsp://admin:***@camera/live"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.1, Fraction(1, 2)),
        (2.5, Fraction(5, 2)),
        (30.0, Fraction(10, 1)),
    ],
)
def test_frame_rate_is_bounded(value: float, expected: Fraction) -> None:
    assert probe._frame_rate(value) == expected


def test_main_reports_probe_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        probe,
        "_run_probe",
        lambda args: {
            "ok": True,
            "decoder_requested": args.decoder,
            "frames": 10,
        },
    )

    assert probe.main(["--test-source", "--decoder", "va"]) == 0
    assert '"frames": 10' in capsys.readouterr().out
