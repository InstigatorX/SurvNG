from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
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


def render_device_available() -> bool:
    """True only when a render node exists. The default path is not proof."""
    root = Path("/dev/dri")
    if not root.is_dir():
        return False
    return any(root.glob("renderD*"))


_hwaccel_cache: dict[str, frozenset[str]] = {}
_hwaccel_lock = threading.Lock()


def ffmpeg_hwaccels(ffmpeg_path: str) -> frozenset[str]:
    """Cached ``ffmpeg -hwaccels`` names. An unreadable binary yields nothing."""
    key = ffmpeg_path or "ffmpeg"
    with _hwaccel_lock:
        cached = _hwaccel_cache.get(key)
        if cached is not None:
            return cached
        names = _probe_ffmpeg_hwaccels(key)
        _hwaccel_cache[key] = names
        return names


def _probe_ffmpeg_hwaccels(ffmpeg_path: str) -> frozenset[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    names: set[str] = set()
    for line in (result.stdout or "").splitlines():
        token = line.strip().lower()
        if not token or " " in token or token.startswith("hardware acceleration"):
            continue
        names.add(token)
    return frozenset(names)


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


@dataclass(frozen=True, slots=True)
class CaptureDecodePlan:
    """One live-capture decoder attempt. ``cpu`` is the software bgr24 path."""

    name: str
    input_args: tuple[str, ...]
    download_filters: tuple[str, ...]


def capture_decode_plan_names(
    mode: str | None,
    qsv_ready: bool,
    vaapi_ready: bool,
) -> tuple[str, ...]:
    """Live capture order is QSV, then VAAPI, then software.

    Recorded-frame decode stays software for ``auto``. This helper is only
    the persistent live/substream capture process.
    """
    selected = hardware_mode(mode)
    names: list[str] = []
    if selected == "off":
        return ("cpu",)
    if selected in {"auto", "qsv"} and qsv_ready:
        names.append("qsv")
    if selected in {"auto", "vaapi"} and vaapi_ready:
        names.append("vaapi")
    names.append("cpu")
    return tuple(names)


def capture_decode_plans(
    mode: str | None,
    *,
    qsv_ready: bool,
    vaapi_ready: bool,
    device: str,
) -> tuple[CaptureDecodePlan, ...]:
    plans: list[CaptureDecodePlan] = []
    for name in capture_decode_plan_names(mode, qsv_ready, vaapi_ready):
        if name == "qsv":
            plans.append(
                CaptureDecodePlan(
                    "qsv",
                    (
                        "-qsv_device",
                        device,
                        "-hwaccel",
                        "qsv",
                        "-hwaccel_output_format",
                        "qsv",
                    ),
                    ("hwdownload", "format=nv12"),
                )
            )
        elif name == "vaapi":
            plans.append(
                CaptureDecodePlan(
                    "vaapi",
                    (
                        "-vaapi_device",
                        device,
                        "-hwaccel",
                        "vaapi",
                        "-hwaccel_output_format",
                        "vaapi",
                    ),
                    ("hwdownload", "format=nv12"),
                )
            )
        else:
            plans.append(CaptureDecodePlan("cpu", (), ()))
    return tuple(plans)


def resolve_capture_decode_plans(
    mode: str | None,
    ffmpeg_path: str,
) -> tuple[CaptureDecodePlan, ...]:
    """Plans for one open. ``off`` and a missing render node never probe FFmpeg."""
    if hardware_mode(mode) == "off" or not render_device_available():
        return capture_decode_plans(
            mode,
            qsv_ready=False,
            vaapi_ready=False,
            device="",
        )
    accels = ffmpeg_hwaccels(ffmpeg_path)
    return capture_decode_plans(
        mode,
        qsv_ready="qsv" in accels,
        vaapi_ready="vaapi" in accels,
        device=dri_render_device(),
    )
