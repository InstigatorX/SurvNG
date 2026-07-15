from __future__ import annotations

from pathlib import Path


def hardware_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    return mode if mode in {"auto", "vaapi", "qsv", "off"} else "auto"


def dri_render_device(default: str = "/dev/dri/renderD128") -> str:
    root = Path("/dev/dri")
    if root.exists():
        devices = sorted(root.glob("renderD*"))
        if devices:
            return str(devices[0])
    return default


def qsv_enabled(value: str | None) -> bool:
    return hardware_mode(value) == "qsv"


def vaapi_enabled(value: str | None) -> bool:
    return hardware_mode(value) == "vaapi"


def qsv_decode_args(value: str | None) -> list[str]:
    if not qsv_enabled(value):
        return []
    return [
        "-qsv_device",
        dri_render_device(),
        "-hwaccel",
        "qsv",
        "-hwaccel_output_format",
        "qsv",
    ]


def recorded_frame_hw_args(value: str | None) -> tuple[list[str], list[str]]:
    if qsv_enabled(value):
        return qsv_decode_args(value), ["-vf", "hwdownload,format=nv12"]
    if vaapi_enabled(value):
        device = dri_render_device()
        return ["-vaapi_device", device, "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"], ["-vf", "hwdownload,format=nv12"]
    return [], []
