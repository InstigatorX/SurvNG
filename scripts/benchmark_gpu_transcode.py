#!/usr/bin/env python3
"""Run every export, timelapse and event-clip attempt on one real recording.

SurvNG stops at the first attempt that works, so a normal export never shows
whether the full-GPU command works or what it saves. This runs each attempt
of the production command builders in turn (full GPU, hardware encode only,
software) on the host's render node and reports wall time, FFmpeg CPU time
and output frames.

Nothing is written outside a temporary directory, and the running service is
not contacted. It does compete with the service for the GPU and CPU.

    .venv/bin/python scripts/benchmark_gpu_transcode.py /path/to/segment.mp4

Pass several consecutive closed segments of one camera for a longer input.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from survng.app.media_exports import MediaExportManager  # noqa: E402
from survng.app.recording_media_runtime import RecordingMediaRuntime  # noqa: E402


class _Captured(Exception):
    """Carries the attempts out of a builder before it runs any of them."""

    def __init__(self, commands: list[tuple[str, list[str]]]) -> None:
        super().__init__("captured")
        self.commands = commands


class _Recorder:
    def lease_recordings_for_playback(self, rows: list[dict], *, ttl_seconds: float) -> None:
        del rows, ttl_seconds


def _ffprobe(ffmpeg: str) -> str:
    sibling = Path(ffmpeg).with_name("ffprobe")
    return str(sibling) if Path(ffmpeg).name == "ffmpeg" and sibling.is_file() else "ffprobe"


def _media_info(ffprobe: str, path: Path) -> dict[str, float | int | str]:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,pix_fmt,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ffprobe failed").strip()[-300:])
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    return {
        "codec": str(stream.get("codec_name") or ""),
        "pix_fmt": str(stream.get("pix_fmt") or ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frames": int(stream.get("nb_frames") or 0),
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
    }


def _run(command: list[str], timeout: float) -> tuple[int, float, float, str]:
    """Return ``(returncode, wall_seconds, cpu_seconds, stderr_tail)``."""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr:
        # Attempts run one at a time, so the growth in reaped-children CPU
        # time across this wait belongs to this FFmpeg process.
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = -9
            stderr.write(f"\nkilled after the {timeout:g} s limit")
        wall = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        stderr.seek(0)
        return code, wall, cpu, stderr.read().strip()[-300:]


def _measure(
    title: str,
    commands: list[tuple[str, list[str]]],
    *,
    ffprobe: str,
    source_seconds: float,
    timeout: float,
    show_commands: bool,
) -> list[dict[str, object]]:
    print(f"\n== {title}")
    print(f"{'attempt':<22}{'result':<8}{'wall s':>8}{'cpu s':>8}{'x real':>8}{'frames':>8}  output")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for name, command in commands:
        if name in seen:
            continue
        seen.add(name)
        output = Path(command[-1])
        output.unlink(missing_ok=True)
        code, wall, cpu, error = _run(command, timeout)
        row: dict[str, object] = {"job": title, "attempt": name, "ok": False, "wall": wall, "cpu": cpu}
        detail = error.splitlines()[-1] if error else f"exit {code}"
        frames = 0
        if code == 0 and output.is_file() and output.stat().st_size > 1024:
            try:
                info = _media_info(ffprobe, output)
                frames = int(info["frames"])
                detail = f"{info['codec']} {info['width']}x{info['height']} {info['pix_fmt']}"
                row["ok"] = frames > 0
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                detail = f"unreadable output: {exc}"
        row["frames"] = frames
        speed = f"{source_seconds / wall:.1f}" if row["ok"] and wall > 0 else "-"
        print(
            f"{name:<22}{'ok' if row['ok'] else 'FAILED':<8}{wall:>8.2f}{cpu:>8.2f}"
            f"{speed:>8}{frames:>8}  {detail}"
        )
        if show_commands or not row["ok"]:
            print("    " + " ".join(command))
        rows.append(row)
        output.unlink(missing_ok=True)
    # SurvNG discards a full-GPU result that lost more than a tenth of its frames.
    reference = next((int(row["frames"]) for row in rows if row["ok"] and str(row["attempt"]).startswith("cpu")), 0)
    for row in rows:
        if row["ok"] and str(row["attempt"]).endswith("-gpu") and int(row["frames"]) < reference * 0.9:
            row["ok"] = False
            print(f"{row['attempt']}: FAILED, {row['frames']} frames against {reference} from cpu; "
                  "SurvNG would discard this output")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path, help="closed recording segment(s) of one camera, in order")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg binary SurvNG is configured to use")
    parser.add_argument("--device", default="", help="render node (default: first /dev/dri/renderD*)")
    parser.add_argument("--backends", default="vaapi,qsv", help="comma-separated: vaapi, qsv")
    parser.add_argument("--height", type=int, default=1080, help="export height; 0 keeps the source size")
    parser.add_argument("--seconds", type=float, default=30.0, help="source seconds for the export and clip jobs")
    parser.add_argument("--interval", type=float, default=30.0, help="timelapse sample interval in seconds")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-attempt limit in seconds")
    parser.add_argument("--show-commands", action="store_true", help="print every command, not only failed ones")
    args = parser.parse_args()

    ffprobe = _ffprobe(args.ffmpeg)
    backends = [name for name in (item.strip() for item in args.backends.split(",")) if name in {"vaapi", "qsv"}]
    devices = sorted(str(path) for path in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    device = args.device or (devices[0] if devices else "")

    sources = [path.resolve() for path in args.recordings]
    infos = [_media_info(ffprobe, path) for path in sources]
    total = sum(float(info["duration"]) for info in infos)
    first = infos[0]
    version = subprocess.run(
        [args.ffmpeg, "-version"], stdout=subprocess.PIPE, text=True, timeout=15,
    ).stdout.splitlines()[0]
    print(version)
    print(f"render nodes: {', '.join(devices) or 'none'}   using: {device or 'none'}")
    print(
        f"source: {len(sources)} file(s), {total:.1f} s, {first['codec']} "
        f"{first['width']}x{first['height']} {first['pix_fmt']}"
    )
    if not device:
        print("No render node: every hardware attempt below is expected to fail.")

    with tempfile.TemporaryDirectory(prefix="survng-gpu-bench-") as temporary:
        root = Path(temporary)
        work = root / "work"
        work.mkdir()

        # The probes SurvNG runs to decide which backend `auto` resolves to.
        runtime = RecordingMediaRuntime(SimpleNamespace(
            get_config=lambda: SimpleNamespace(ffmpeg_path=args.ffmpeg, hardware_acceleration="auto"),
        ))
        if device:
            # Probe the node the attempts run on, not always the first one.
            runtime._dri_render_devices = lambda: [device]  # type: ignore[method-assign]
        available: dict[str, bool] = {}
        for name, probe in (("vaapi", runtime._ffmpeg_vaapi_info), ("qsv", runtime._ffmpeg_qsv_info)):
            info = probe()
            available[name] = bool(info.get("available"))
            reason = "" if available[name] else f"  ({str(info.get('runtime_error') or 'not listed by FFmpeg')[-160:]})"
            print(f"SurvNG {name} probe: {'usable' if available[name] else 'unavailable'}{reason}")
        print(f"auto resolves to: {runtime._media_export_hardware_backend()}")

        rows: list[dict] = []
        epoch = 1_000_000.0
        for path, info in zip(sources, infos):
            duration = float(info["duration"])
            rows.append({
                "path": str(path), "start_epoch": epoch, "end_epoch": epoch + duration,
                "duration_seconds": duration,
                # SurvNG only keeps frames on the GPU across recordings its
                # index knows to share one stream layout.
                "stream_fingerprint": f"{info['codec']}:{info['width']}x{info['height']}:{info['pix_fmt']}",
            })
            epoch += duration
        clip_seconds = max(1.0, min(args.seconds, total))
        results: list[dict[str, object]] = []

        def captured(backend: str, kind: str, options: dict, end_epoch: float, *, keyframes: bool = True):
            manager = MediaExportManager(
                root / f"storage-{backend}", root / f"database-{backend}",
                recorder=lambda: _Recorder(), ffmpeg_path=lambda: args.ffmpeg,
                hardware_backend=lambda: backend,
                hardware_device=lambda _backend: device,
            )

            def capture(*call_args, **call_kwargs):
                builder = original_recording if kind == "recording" else original_timelapse
                raise _Captured(builder(*call_args, **call_kwargs))

            original_recording = manager._recording_commands
            original_timelapse = manager._timelapse_commands
            manager._recording_commands = capture  # type: ignore[method-assign]
            manager._timelapse_commands = capture  # type: ignore[method-assign]
            spacing = manager._keyframe_interval(sources[0])
            if not keyframes:
                manager._keyframe_interval = lambda _path: None  # type: ignore[method-assign]
            job = manager.store.create({
                "kind": kind, "camera_id": "bench", "source": "main",
                "start_epoch": rows[0]["start_epoch"], "end_epoch": end_epoch, "options": options,
            })
            build = manager._build_recording if kind == "recording" else manager._build_timelapse
            try:
                build(job, rows, work, threading.Event())
            except _Captured as result:
                return result.commands, spacing
            raise RuntimeError(f"{kind} builder produced no commands")

        def merged(kind: str, options: dict, end_epoch: float, **kwargs):
            commands: list[tuple[str, list[str]]] = []
            spacing = None
            for backend in [*backends, "cpu"]:
                attempts, spacing = captured(backend, kind, options, end_epoch, **kwargs)
                commands.extend(attempt for attempt in attempts if backend == "cpu" or attempt[0] != "cpu")
            return commands, spacing

        measure = dict(ffprobe=ffprobe, timeout=args.timeout, show_commands=args.show_commands)

        commands, _ = merged("recording", {"height": args.height}, rows[0]["start_epoch"] + clip_seconds)
        if not any(name.endswith("-gpu") for name, _ in commands):
            print("\nNo full-GPU export attempt: the source does not exactly fill the output frame "
                  "(software pad needed) or its codec has no QSV decoder. Try --height 0.")
        results += _measure(
            f"recording export, {clip_seconds:.0f} s to height {args.height or 'source'}",
            commands, source_seconds=clip_seconds, **measure,
        )

        concat = MediaExportManager._write_concat(rows, work / "clip.ffconcat")
        clip = work / "clip.mp4"
        codec = str(first["codec"])
        clip_commands: list[tuple[str, list[str]]] = []
        if "vaapi" in backends and codec in {"h264", "hevc"}:
            runtime._ffmpeg_vaapi_info = lambda: {"device": device}  # type: ignore[method-assign]
            # SurvNG skips hardware decode for 10-bit sources.
            if "10" not in str(first["pix_fmt"]) and "12" not in str(first["pix_fmt"]):
                clip_commands.append(("vaapi", runtime._event_clip_vaapi_command(codec, concat, 0.0, clip_seconds, clip)))
            clip_commands.append(("vaapi-encode", runtime._event_clip_vaapi_command(
                codec, concat, 0.0, clip_seconds, clip, hardware_decode=False)))
        if "qsv" in backends and codec in {"h264", "hevc"}:
            runtime._ffmpeg_qsv_info = lambda: {"render_devices": [device] if device else []}  # type: ignore[method-assign]
            clip_commands.append(("qsv", runtime._event_clip_qsv_command(codec, concat, 0.0, clip_seconds, clip)))
        clip_commands.append(("cpu", runtime._event_clip_cpu_command(concat, 0.0, clip_seconds, clip)))
        results += _measure(
            f"event clip, {clip_seconds:.0f} s", clip_commands, source_seconds=clip_seconds, **measure,
        )

        # Height 0 means source size; the timelapse expresses that as a width.
        options = {"sample_interval_seconds": args.interval, **(
            {"height": args.height} if args.height > 0 else {"width": int(first["width"])}
        )}
        commands, _ = merged("timelapse", options, epoch, keyframes=False)
        results += _measure(
            f"timelapse, every frame decoded, 1 frame per {args.interval:g} s",
            commands, source_seconds=total, **measure,
        )
        commands, spacing = merged("timelapse", options, epoch)
        if any("-skip_frame" in command for _name, command in commands):
            results += _measure(
                f"timelapse, keyframes only (spacing {spacing:.2f} s)",
                [(f"{name}-keyframes", command) for name, command in commands if "-skip_frame" in command],
                source_seconds=total, **measure,
            )
        else:
            known = "unknown" if spacing is None else f"{spacing:.2f} s"
            print(f"\nKeyframe-only timelapse not used: keyframe spacing is {known}, and "
                  f"--interval {args.interval:g} must be at least ten times that.")

    # A backend SurvNG's own probe calls usable must run every attempt built for it.
    failures = [
        row for row in results
        if not row["ok"] and available.get(str(row["attempt"]).split("-")[0], str(row["attempt"]).startswith("cpu"))
    ]
    print()
    for job in dict.fromkeys(str(row["job"]) for row in results):
        ok = [row for row in results if row["job"] == job and row["ok"]]
        cpu = next((row for row in ok if str(row["attempt"]).startswith("cpu")), None)
        best = min(ok, key=lambda row: float(row["cpu"]), default=None)  # type: ignore[arg-type]
        if cpu and best and best is not cpu and float(best["cpu"]) > 0:
            print(f"{job}: {best['attempt']} used {float(cpu['cpu']) / float(best['cpu']):.1f}x less "
                  f"FFmpeg CPU than cpu ({float(best['cpu']):.2f} s vs {float(cpu['cpu']):.2f} s)")
    every = next((row for row in results if row["ok"] and row["attempt"] == "cpu"
                  and "every frame" in str(row["job"])), None)
    keyed = next((row for row in results if row["ok"] and row["attempt"] == "cpu-keyframes"), None)
    if every and keyed and float(keyed["cpu"]) > 0:
        print(f"timelapse: keyframe-only decode used {float(every['cpu']) / float(keyed['cpu']):.1f}x less "
              f"FFmpeg CPU than decoding every frame ({float(keyed['cpu']):.2f} s vs {float(every['cpu']):.2f} s), "
              f"{keyed['frames']} vs {every['frames']} output frames")
    if failures:
        print("FAILED attempts on a usable backend: "
              + ", ".join(f"{row['attempt']} [{row['job']}]" for row in failures))
        return 1
    print("Every attempt on a usable backend produced playable video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
