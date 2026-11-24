"""Recording media runtime ownership: caching, remux, previews, clips, exports, and hardware probes."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
from fastapi import HTTPException
from fastapi.responses import FileResponse

from .config import AppConfig, slugify_camera_id
from .encoded_fragments import IndexedMp4FragmentSource
from .ffmpeg_hw import RECORDED_FRAME_INPUT_THREAD_ARGS, RECORDED_FRAME_OUTPUT_THREAD_ARGS
from .incident_utils import event_epoch
from .manager import AppManager
from .media_exports import MediaExportManager
from .media_sessions import (
    MediaResourceClass,
    MediaSessionAdmissionError,
    MediaSessionKind,
    MediaSessionLease,
    MediaSessionRequest,
)
from .recording_media import RECORDING_FMP4_VERSION, concatenated_clip_timing, event_clip_window, mp4_video_dimensions, playback_segment_duration
from .recording_mp4 import offset_fmp4_timestamps
from .recorded_decoder import (
    FAILURE_TIMEOUT,
    DecodePlan,
    build_decode_command,
    read_bmp_frames,
    recorded_decode_plans,
    run_decoder,
    warn_decoder_errors,
    warn_decoder_failure,
)
from .recording_frame_steps import (
    FRAME_TIME_TOLERANCE_SECONDS,
    SegmentFrames,
    frame_index_at,
    frame_run,
    parse_packet_listing,
    scaled_size,
    showinfo_times,
)
from .recording_routes import recording_source
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)

# Scrub-preview strips: tiles per row, the most tiles one segment gets, and
# the frame size (about 1280x800) up to which every frame is decoded.
RECORDING_STRIP_COLUMNS = 6
RECORDING_STRIP_MAX_TILES = 60
RECORDING_STRIP_FULL_DECODE_PIXELS = 1_024_000
# After a failed GPU strip, how long strips are made on the CPU.
RECORDING_STRIP_HARDWARE_RETRY_SECONDS = 300.0
# A GPU strip takes a fraction of a second; one that takes this long is hung.
RECORDING_STRIP_HARDWARE_TIMEOUT_SECONDS = 5.0
# Frame stepping: segments whose frame lists are remembered, the budget for
# decoding one run, and the JPEG quality of a stepped frame (it is looked at
# for detail).
RECORDING_FRAME_LIST_CACHE_ENTRIES = 64
RECORDING_FRAME_RUN_TIMEOUT_SECONDS = 10.0
RECORDING_FRAME_JPEG_QUALITY = 88


class RecordingPrewarmCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LeasedRecordingSegment:
    path: Path
    lease_token: str


@dataclass(frozen=True, slots=True)
class RecordingMediaDependencies:
    get_config: Callable[[], AppConfig]
    get_manager: Callable[[], AppManager]
    ffprobe_path: Callable[[], str]
    validate_recording_range: Callable[[float, float, float, str], None]
    recording_playback_window: Callable[[float], tuple[float, float]]


class RecordingMediaRuntime:
    # Timeline scrubbing can mint many distinct day-window keys; TTL alone does
    # not bound peak memory under long-uptime interactive use.
    RECORDING_DAY_CACHE_MAX_ENTRIES = 256

    def __init__(self, deps: RecordingMediaDependencies) -> None:
        self.deps = deps
        self.recording_fmp4_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
        self.recording_fmp4_locks_guard = threading.Lock()
        self.event_clip_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
        self.event_clip_locks_guard = threading.Lock()
        self.recording_day_cache: dict[tuple[str, str, int, int], tuple[float, list[dict]]] = {}
        self.recording_day_cache_lock = threading.Lock()
        self.recording_day_cache_seconds = 30.0
        self.recording_near_live_cache_seconds = 2.0
        self.recording_day_cache_max_entries = self.RECORDING_DAY_CACHE_MAX_ENTRIES
        self.recording_preview_interval_seconds = 5.0
        self.recording_preview_max_age_seconds = 7 * 24 * 60 * 60
        self.recording_preview_max_bytes = 256 * 1024 * 1024
        self.recording_preview_build_limiter = threading.BoundedSemaphore(1)
        # Strips are asked for in pairs (where the playhead is, and the
        # segment it is heading into). Each build also holds a transcode
        # media session, of which there are two.
        self.recording_strip_build_limiter = threading.BoundedSemaphore(2)
        self.recording_strip_hardware_retry_at = 0.0
        # Frame-step runs: one decode at a time, apart from the single
        # preview build that incident and training stills wait on.
        self.recording_frame_run_build_limiter = threading.BoundedSemaphore(1)
        self.recording_frame_lists: OrderedDict[tuple[str, int, int], SegmentFrames] = OrderedDict()
        self.recording_frame_list_lock = threading.Lock()
        self.recording_preview_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
        self.recording_preview_locks_guard = threading.Lock()
        self.recording_preview_maintenance_lock = threading.Lock()
        self.recording_preview_last_maintenance = 0.0
        self.recording_cache_maintenance_lock = threading.Lock()
        self.recording_cache_last_maintenance = 0.0
        self.recording_cache_status_lock = threading.Lock()
        self.recording_cache_status_cached_at = 0.0
        self.recording_cache_status_cached_inventory = (0, 0)
        self.recording_cache_status_seconds = 5.0
        self.recording_cache_metrics_lock = threading.Lock()
        self.recording_cache_metrics = {
            "playback_hits": 0.0, "playback_misses": 0.0,
            "playback_remuxes": 0.0, "playback_failures": 0.0,
            "playback_remux_ms": 0.0, "playback_last_remux_ms": 0.0,
            "prewarm_hits": 0.0, "prewarm_misses": 0.0,
            "prewarm_remuxes": 0.0, "prewarm_failures": 0.0,
            "prewarm_remux_ms": 0.0, "prewarm_last_remux_ms": 0.0,
        }
        self.recording_prewarm_stop = threading.Event()
        self.recording_prewarm_thread: threading.Thread | None = None
        self.recording_prewarm_process_lock = threading.Lock()
        self.recording_prewarm_process: subprocess.Popen | None = None
        self.event_clip_build_limiter = threading.BoundedSemaphore(2)
        self.media_exports_lock = threading.Lock()
        self.media_exports: MediaExportManager | None = None
        self._qsv_cache: tuple[tuple[str, tuple[str, ...]], dict] | None = None
        self._vaapi_cache: tuple[tuple[str, tuple[str, ...]], dict] | None = None
        self._hardware_probe_lock = threading.Lock()

    @property
    def config(self) -> AppConfig:
        return self.deps.get_config()

    @property
    def manager(self) -> AppManager:
        return self.deps.get_manager()

    def clear_runtime_caches(self) -> None:
        with self.recording_day_cache_lock:
            self.recording_day_cache.clear()
        with self.recording_cache_status_lock:
            self.recording_cache_status_cached_at = 0.0
            self.recording_cache_status_cached_inventory = (0, 0)
        self.clear_hardware_probe_caches()

    def encoded_fragment_source(
        self,
        active_manager: AppManager | None = None,
    ) -> IndexedMp4FragmentSource:
        selected_manager = active_manager or self.manager
        return IndexedMp4FragmentSource(
            row_loader=lambda camera_id, start_epoch, end_epoch, source: (
                self._recording_day_rows(
                    camera_id,
                    start_epoch,
                    end_epoch,
                    source,
                    fresh=True,
                    active_manager=selected_manager,
                )
            ),
            tolerant_row_loader=lambda camera_id, start_epoch, end_epoch, source: (
                self._recording_day_rows(
                    camera_id,
                    start_epoch,
                    end_epoch,
                    source,
                    fresh=True,
                    active_manager=selected_manager,
                    keep_unknown=True,
                )
            ),
            path_resolver=lambda value: self._recording_storage_path(
                value,
                active_manager=selected_manager,
            ),
            # The resolver reports a missing or out-of-storage file as an
            # HTTP error; a tolerant listing treats it as unavailable.
            unavailable_errors=(HTTPException,),
            remuxer=lambda path, duration, media_offset: (
                self._recording_fmp4_files(
                    path,
                    duration,
                    media_offset,
                    active_manager=selected_manager,
                )
            ),
        )

    def clear_hardware_probe_caches(self) -> None:
        with self._hardware_probe_lock:
            self._qsv_cache = None
            self._vaapi_cache = None

    def prewarmer_running(self) -> bool:
        return bool(self.recording_prewarm_thread and self.recording_prewarm_thread.is_alive())

    def active_export_jobs(self) -> list[dict]:
        return self.media_exports.active_jobs() if self.media_exports is not None else []

    def cache_status(self) -> dict:
        now = time.monotonic()
        with self.recording_cache_status_lock:
            if now - self.recording_cache_status_cached_at >= self.recording_cache_status_seconds:
                self.recording_cache_status_cached_inventory = self._recording_cache_inventory()
                self.recording_cache_status_cached_at = now
            entries, total_bytes = self.recording_cache_status_cached_inventory
        with self.recording_cache_metrics_lock:
            metrics = dict(self.recording_cache_metrics)
        for origin in ("playback", "prewarm"):
            remuxes = int(metrics[f"{origin}_remuxes"])
            metrics[f"{origin}_avg_remux_ms"] = round(float(metrics[f"{origin}_remux_ms"]) / remuxes, 1) if remuxes else 0.0
            metrics[f"{origin}_last_remux_ms"] = round(float(metrics[f"{origin}_last_remux_ms"]), 1)
            metrics.pop(f"{origin}_remux_ms", None)
        return {
            "entries": entries,
            "bytes": total_bytes,
            "max_bytes": int(float(self.config.recording_cache_max_gb) * 1024 * 1024 * 1024),
            "max_days": int(self.config.recording_cache_max_days),
            "prewarm": bool(self.config.recording_cache_prewarm),
            "metrics": metrics,
            "media_sessions": self.manager.media_session_status(),
        }

    def _recording_cache_inventory(self) -> tuple[int, int]:
        root = self.manager.storage_dir / "playback-cache" / "fmp4"
        if not root.exists():
            return (0, 0)
        entries = 0
        total_bytes = 0
        try:
            cache_directories = os.scandir(root)
        except OSError:
            return (0, 0)
        with cache_directories:
            for directory in cache_directories:
                try:
                    if not directory.is_dir(follow_symlinks=False):
                        continue
                    has_files = False
                    with os.scandir(directory.path) as files:
                        for file_entry in files:
                            if not file_entry.is_file(follow_symlinks=False):
                                continue
                            try:
                                total_bytes += file_entry.stat(follow_symlinks=False).st_size
                                has_files = True
                            except OSError:
                                continue
                    if has_files:
                        entries += 1
                except OSError:
                    continue
        return entries, total_bytes


    def _run_ffmpeg_list(
        self,
        args: list[str],
        timeout: float = 5.0,
        *,
        ffmpeg_path: str | None = None,
    ) -> str:
        try:
            result = subprocess.run([ffmpeg_path or self.config.ffmpeg_path, '-hide_banner', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return result.stdout or ''
        except Exception:
            return ''

    def _dri_render_devices(self) -> list[str]:
        return sorted((str(path) for path in Path('/dev/dri').glob('renderD*'))) if Path('/dev/dri').exists() else []

    def _ffmpeg_qsv_info(self) -> dict:
        render_devices = self._dri_render_devices()
        ffmpeg_path = self.config.ffmpeg_path
        cache_key = (ffmpeg_path, tuple(render_devices))
        with self._hardware_probe_lock:
            if self._qsv_cache is not None and self._qsv_cache[0] == cache_key:
                return self._qsv_cache[1]
            hwaccels = self._run_ffmpeg_list(['-hwaccels'], ffmpeg_path=ffmpeg_path)
            encoders = self._run_ffmpeg_list(['-encoders'], ffmpeg_path=ffmpeg_path)
            decoders = self._run_ffmpeg_list(['-decoders'], ffmpeg_path=ffmpeg_path)
            qsv_encoders = sorted({name for name in ('h264_qsv', 'hevc_qsv', 'av1_qsv', 'mjpeg_qsv') if name in encoders})
            qsv_decoders = sorted({name for name in ('h264_qsv', 'hevc_qsv', 'av1_qsv', 'mjpeg_qsv') if name in decoders})
            listed = 'qsv' in hwaccels and 'h264_qsv' in encoders
            runtime_usable = False
            runtime_error = ''
            if listed:
                probe_args = [ffmpeg_path, '-hide_banner', '-v', 'error']
                if render_devices:
                    probe_args.extend(['-qsv_device', render_devices[0]])
                probe_args.extend(['-f', 'lavfi', '-i', 'color=size=64x64:rate=1', '-frames:v', '1', '-c:v', 'h264_qsv', '-f', 'null', '-'])
                try:
                    probe = subprocess.run(probe_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=8)
                    runtime_usable = probe.returncode == 0
                    runtime_error = '' if runtime_usable else (probe.stderr or 'QSV runtime probe failed').strip()[-500:]
                except Exception as exc:
                    runtime_error = str(exc) or 'QSV runtime probe failed'
            result = {'available': bool(listed and runtime_usable), 'listed': bool(listed), 'runtime_usable': runtime_usable, 'runtime_error': runtime_error, 'hwaccel_listed': 'qsv' in hwaccels, 'encoders': qsv_encoders, 'decoders': qsv_decoders, 'render_devices': render_devices}
            self._qsv_cache = (cache_key, result)
            return result

    def _ffmpeg_vaapi_info(self) -> dict:
        render_devices = self._dri_render_devices()
        ffmpeg_path = self.config.ffmpeg_path
        cache_key = (ffmpeg_path, tuple(render_devices))
        with self._hardware_probe_lock:
            if self._vaapi_cache is not None and self._vaapi_cache[0] == cache_key:
                return self._vaapi_cache[1]
            hwaccels = self._run_ffmpeg_list(['-hwaccels'], ffmpeg_path=ffmpeg_path)
            encoders = self._run_ffmpeg_list(['-encoders'], ffmpeg_path=ffmpeg_path)
            decoders = self._run_ffmpeg_list(['-decoders'], ffmpeg_path=ffmpeg_path)
            filters = self._run_ffmpeg_list(['-filters'], ffmpeg_path=ffmpeg_path)
            vaapi_encoders = sorted({name for name in ('h264_vaapi', 'hevc_vaapi', 'av1_vaapi', 'mjpeg_vaapi', 'mpeg2_vaapi', 'vp8_vaapi', 'vp9_vaapi') if name in encoders})
            vaapi_decoders = sorted({name for name in ('h264_vaapi', 'hevc_vaapi', 'av1_vaapi', 'mjpeg_vaapi', 'mpeg2_vaapi', 'vp8_vaapi', 'vp9_vaapi') if name in decoders})
            vaapi_filters = sorted({name for name in ('hwupload', 'scale_vaapi') if name in filters})
            listed = 'vaapi' in hwaccels and 'h264_vaapi' in encoders and ('hwupload' in filters)
            runtime_usable = False
            runtime_error = ''
            if listed and render_devices:
                probe_args = [ffmpeg_path, '-hide_banner', '-v', 'error', '-vaapi_device', render_devices[0], '-f', 'lavfi', '-i', 'color=size=64x64:rate=1', '-frames:v', '1', '-vf', 'format=nv12,hwupload', '-c:v', 'h264_vaapi', '-f', 'null', '-']
                try:
                    probe = subprocess.run(probe_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=8)
                    runtime_usable = probe.returncode == 0
                    runtime_error = '' if runtime_usable else (probe.stderr or 'VAAPI runtime probe failed').strip()[-500:]
                except Exception as exc:
                    runtime_error = str(exc) or 'VAAPI runtime probe failed'
            elif listed:
                runtime_error = 'No /dev/dri/renderD* render device found'
            result = {'available': bool(listed and runtime_usable), 'listed': bool(listed), 'runtime_usable': runtime_usable, 'runtime_error': runtime_error, 'hwaccel_listed': 'vaapi' in hwaccels, 'encoders': vaapi_encoders, 'decoders': vaapi_decoders, 'filters': vaapi_filters, 'render_devices': render_devices, 'device': render_devices[0] if render_devices else ''}
            self._vaapi_cache = (cache_key, result)
            return result

    def _hardware_acceleration_mode(self) -> str:
        mode = str(getattr(self.config, 'hardware_acceleration', 'auto') or 'auto').lower()
        return mode if mode in {'auto', 'vaapi', 'qsv', 'off'} else 'auto'

    def _media_export_hardware_backend(self) -> str:
        """Resolve the configured, currently usable H.264 export encoder."""
        mode = self._hardware_acceleration_mode()
        if mode == 'off':
            return 'cpu'
        if mode in {'auto', 'vaapi'}:
            info = self._ffmpeg_vaapi_info()
            if info.get('available') and 'h264_vaapi' in set(info.get('encoders') or []):
                return 'vaapi'
            if mode == 'vaapi':
                return 'cpu'
        if mode in {'auto', 'qsv'}:
            info = self._ffmpeg_qsv_info()
            if info.get('available') and 'h264_qsv' in set(info.get('encoders') or []):
                return 'qsv'
        return 'cpu'

    def _media_export_hardware_device(self, backend: str) -> str:
        info = self._ffmpeg_qsv_info() if backend == 'qsv' else self._ffmpeg_vaapi_info()
        devices = info.get('render_devices') or []
        return str(devices[0]) if devices else str(info.get('device') or '')

    def _media_export_manager(self) -> MediaExportManager:
        active_manager = self.manager
        with self.media_exports_lock:
            if self.media_exports is None:
                self.media_exports = self._new_media_export_manager(active_manager)
            selected = self.media_exports
            correctly_bound = (
                selected.storage_dir == active_manager.storage_dir.resolve()
                and selected.database_dir == active_manager.database_dir.resolve()
                and selected.media_storage is active_manager.media_storage
            )
        if not correctly_bound:
            if not self.rebind_media_exports(active_manager=active_manager):
                raise HTTPException(
                    status_code=503,
                    detail='media exports are unavailable while storage is being reconfigured',
                )
            with self.media_exports_lock:
                selected = self.media_exports
        return selected

    def _new_media_export_manager(
        self, active_manager: AppManager | None = None
    ) -> MediaExportManager:
        selected_manager = active_manager or self.manager
        return MediaExportManager(
            storage_dir=selected_manager.storage_dir,
            database_dir=selected_manager.database_dir,
            recorder=lambda: selected_manager.recorder,
            ffmpeg_path=lambda: self.config.ffmpeg_path,
            hardware_backend=self._media_export_hardware_backend,
            hardware_device=self._media_export_hardware_device,
            media_storage=selected_manager.media_storage,
            media_sessions=selected_manager.media_sessions,
            owner_generation=selected_manager.media_session_generation,
        )

    def rebind_media_exports(
        self, *, active_manager: AppManager | None = None
    ) -> bool:
        """Move the export worker to the current manager's storage generation."""
        selected_manager = active_manager or self.manager
        with self.media_exports_lock:
            previous = self.media_exports
            if previous is None:
                return True
            if (
                previous.storage_dir == selected_manager.storage_dir.resolve()
                and previous.database_dir == selected_manager.database_dir.resolve()
                and previous.media_storage is selected_manager.media_storage
            ):
                return True
            was_running = previous.is_running()
            if was_running and not previous.stop(timeout=10.0):
                LOGGER.error(
                    'media export worker could not stop during manager cutover; '
                    'retaining its existing storage binding'
                )
                return False
            try:
                replacement = self._new_media_export_manager(selected_manager)
                if was_running:
                    replacement.start()
            except Exception:
                LOGGER.exception(
                    'media export worker could not bind to the replacement manager'
                )
                if was_running:
                    try:
                        previous.start()
                    except Exception:
                        LOGGER.exception('previous media export worker could not restart')
                return False
            self.media_exports = replacement
            return True

    def _probe_video_codec(self, path: Path) -> str:
        try:
            result = subprocess.run([self.deps.ffprobe_path(), '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=8)
            return (result.stdout or '').strip().lower()
        except Exception:
            return ''

    def _event_clip_vaapi_enabled(self, source_codec: str) -> bool:
        mode = self._hardware_acceleration_mode()
        if mode not in {'auto', 'vaapi'}:
            return False
        if source_codec not in {'h264', 'hevc'}:
            return False
        info = self._ffmpeg_vaapi_info()
        has_encoder = 'h264_vaapi' in set(info.get('encoders') or [])
        return bool(info.get('available') and has_encoder)

    def _event_clip_qsv_enabled(self, source_codec: str) -> bool:
        mode = self._hardware_acceleration_mode()
        if mode == 'off':
            return False
        if mode == 'auto' and self._ffmpeg_vaapi_info().get('available'):
            return False
        if mode not in {'auto', 'qsv'}:
            return False
        if source_codec not in {'h264', 'hevc'}:
            return False
        info = self._ffmpeg_qsv_info()
        decoder = f'{source_codec}_qsv'
        has_decoder = decoder in set(info.get('decoders') or [])
        has_encoder = 'h264_qsv' in set(info.get('encoders') or [])
        return bool(info.get('available') and has_decoder and has_encoder)

    def _event_clip_cpu_command(self, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
        return [self.config.ffmpeg_path, '-hide_banner', '-loglevel', 'warning', '-f', 'concat', '-safe', '0', '-i', str(concat_path), '-ss', f'{local_start:.3f}', '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', str(tmp_path)]

    def _event_clip_vaapi_command(self, source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
        info = self._ffmpeg_vaapi_info()
        device = str(info.get('device') or '/dev/dri/renderD128')
        return [self.config.ffmpeg_path, '-hide_banner', '-loglevel', 'warning', '-vaapi_device', device, '-f', 'concat', '-safe', '0', '-i', str(concat_path), '-ss', f'{local_start:.3f}', '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'format=nv12,hwupload', '-c:v', 'h264_vaapi', '-qp', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', str(tmp_path)]

    def _event_clip_qsv_command(self, source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
        decoder = 'hevc_qsv' if source_codec == 'hevc' else 'h264_qsv'
        info = self._ffmpeg_qsv_info()
        render_devices = info.get('render_devices') or []
        device_args = ['-qsv_device', render_devices[0]] if render_devices else []
        return [self.config.ffmpeg_path, '-hide_banner', '-loglevel', 'warning', *device_args, '-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv', '-c:v', decoder, '-f', 'concat', '-safe', '0', '-i', str(concat_path), '-ss', f'{local_start:.3f}', '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0?', '-c:v', 'h264_qsv', '-preset', 'veryfast', '-global_quality', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', str(tmp_path)]

    def _recording_day_rows(self, camera_id: str, start_epoch: float, end_epoch: float, source: str, *, fresh: bool=False, active_manager: AppManager | None=None, keep_unknown: bool=False) -> list[dict]:
        selected_manager = active_manager or self.manager
        selected_source = recording_source(source)
        cache_key = (camera_id, selected_source, int(start_epoch), int(end_epoch))
        now = time.monotonic()
        if not fresh:
            with self.recording_day_cache_lock:
                cached = self.recording_day_cache.get(cache_key)
            near_live = end_epoch >= time.time() - max(30.0, selected_manager.recorder.segment_seconds * 3)
            cache_seconds = self.recording_near_live_cache_seconds if near_live else self.recording_day_cache_seconds
            if cached is not None and now - cached[0] < cache_seconds:
                rows = [dict(row) for row in cached[1]]
                selected_manager.recorder.lease_recordings_for_playback(rows)
                selected_manager.recorder.resolve_recording_playback_metadata(rows)
                return rows
        rows = [row for row in selected_manager.recorder.recording_rows_between(camera_id, start_epoch, end_epoch, selected_source, discover_missing=False) if int(row.get('size_bytes') or 0) > 1024]
        if keep_unknown:
            # A growing playlist must see rows whose file cannot be checked
            # right now (they are marked), not lose them for one re-read.
            rows = selected_manager.recorder.discard_missing_recording_rows(rows, keep_unknown=True)
        elif fresh:
            rows = selected_manager.recorder.discard_missing_recording_rows(rows)
        selected_manager.recorder.lease_recordings_for_playback(rows)
        # The index retains exact video sample durations separately from its
        # availability estimates. Unknown legacy files are resolved once here.
        selected_manager.recorder.resolve_recording_playback_metadata(rows)
        selected_manager.recorder.queue_stream_fingerprints(rows)
        if keep_unknown:
            # Marked rows must never reach the shared row cache.
            return rows
        with self.recording_day_cache_lock:
            self.recording_day_cache[cache_key] = (now, rows)
            expired = [
                key
                for key, value in self.recording_day_cache.items()
                if now - value[0] >= self.recording_day_cache_seconds
            ]
            for key in expired:
                self.recording_day_cache.pop(key, None)
            overflow = len(self.recording_day_cache) - self.recording_day_cache_max_entries
            if overflow > 0:
                oldest = sorted(
                    self.recording_day_cache.items(),
                    key=lambda item: item[1][0],
                )[:overflow]
                for key, _value in oldest:
                    self.recording_day_cache.pop(key, None)
        return rows

    def _recording_cache_metric(self, origin: str, metric: str, value: float=1.0) -> None:
        key = f'{origin}_{metric}'
        with self.recording_cache_metrics_lock:
            self.recording_cache_metrics[key] = float(self.recording_cache_metrics.get(key, 0.0)) + value

    def _signal_recording_prewarm_process(self, process: subprocess.Popen, sig: signal.Signals) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def _run_recording_remux(
        self,
        command: list[str],
        origin: str,
        session: MediaSessionLease | None,
    ) -> subprocess.CompletedProcess:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        if session is not None:
            session.attach_process(process.pid)
        if origin == 'prewarm':
            with self.recording_prewarm_process_lock:
                self.recording_prewarm_process = process
        terminate_at: float | None = None
        timeout_at = time.monotonic() + 30.0
        try:
            while True:
                cancelled = (
                    (session is not None and session.cancelled())
                    or (
                        origin == 'prewarm'
                        and self.recording_prewarm_stop.is_set()
                    )
                )
                if not cancelled and time.monotonic() >= timeout_at:
                    self._signal_recording_prewarm_process(process, signal.SIGTERM)
                    try:
                        _stdout, stderr = process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._signal_recording_prewarm_process(process, signal.SIGKILL)
                        _stdout, stderr = process.communicate()
                    raise subprocess.TimeoutExpired(command, 30, stderr=stderr)
                if cancelled and process.poll() is None:
                    if terminate_at is None:
                        self._signal_recording_prewarm_process(process, signal.SIGTERM)
                        terminate_at = time.monotonic() + 3.0
                    elif time.monotonic() >= terminate_at:
                        self._signal_recording_prewarm_process(process, signal.SIGKILL)
                try:
                    _stdout, stderr = process.communicate(timeout=0.25)
                    if cancelled:
                        raise RecordingPrewarmCancelled
                    return subprocess.CompletedProcess(command, process.returncode, None, stderr)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if origin == 'prewarm':
                with self.recording_prewarm_process_lock:
                    if self.recording_prewarm_process is process:
                        self.recording_prewarm_process = None

    def _recording_fmp4_files(
        self,
        path: Path,
        duration: float,
        media_offset: float,
        origin: str = 'playback',
        *,
        active_manager: AppManager | None = None,
    ) -> tuple[Path, Path]:
        selected_manager = active_manager or self.manager
        selected_config = getattr(selected_manager, 'config', self.config)
        stat = path.stat()
        fingerprint = f'v{RECORDING_FMP4_VERSION}:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}:{media_offset:.3f}'
        cache_key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:24]
        cache_dir = selected_manager.storage_dir / 'playback-cache' / 'fmp4' / cache_key
        init_path = cache_dir / 'init.mp4'
        media_path = cache_dir / 'media.m4s'
        if self._recording_cache_files_ready(init_path, media_path, touch=True):
            self._recording_cache_metric(origin, 'hits')
            return (init_path, media_path)
        with self.recording_fmp4_locks_guard:
            lock = self.recording_fmp4_locks.setdefault(cache_key, threading.Lock())
        with lock:
            if self._recording_cache_files_ready(init_path, media_path, touch=True):
                self._recording_cache_metric(origin, 'hits')
                return (init_path, media_path)
            self._recording_cache_metric(origin, 'misses')
            remux_started = time.monotonic()
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(prefix='fmp4-', dir=cache_dir))
            codec = self._probe_video_codec(path)
            # Remux each source locally, then shift tfdt and sidx once below.
            # FFmpeg also carries output_ts_offset in the init edit lists. With
            # tfdt repair it shifts playback twice when a later segment uses its
            # own init (for example, after a codec change).
            # Preserve decoder preroll too: automatic negative-timestamp shifting
            # adds per-source A/V edit offsets that native HLS can turn into gaps.
            command = [selected_config.ffmpeg_path, '-hide_banner', '-loglevel', 'warning', '-i', str(path), '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-avoid_negative_ts', 'disabled']
            if codec in {'hevc', 'h265'}:
                command.extend(['-tag:v', 'hvc1'])
            # Let native HLS preroll audio and video together after a seek.
            # The MOV muxer's default writes the entire video run before audio.
            command.extend([
                '-f', 'hls', '-hls_time', '300', '-hls_list_size', '0',
                '-hls_segment_type', 'fmp4', '-hls_segment_options', 'frag_interleave=1',
                '-hls_fmp4_init_filename', 'init.mp4',
                '-hls_segment_filename', str(temp_dir / 'media_%d.m4s'),
                str(temp_dir / 'index.m3u8'),
            ])
            try:
                media_sessions = getattr(selected_manager, 'media_sessions', None)
                if media_sessions is None:
                    result = self._run_recording_remux(command, origin, None)
                else:
                    session = media_sessions.acquire(
                        MediaSessionRequest(
                            (
                                MediaSessionKind.RECORDING_PREWARM
                                if origin == 'prewarm'
                                else MediaSessionKind.RECORDING_REMUX
                            ),
                            resources={MediaResourceClass.REMUX_PROCESS: 1},
                            owner_generation=getattr(
                                selected_manager,
                                'media_session_generation',
                                None,
                            ),
                        ),
                        blocking=origin != 'prewarm',
                        # HLS init/media requests are required dependencies.
                        # Queue cold playback behind bounded remux work rather
                        # than turning transient contention into a failed seek.
                        timeout=None,
                    )
                    with session:
                        result = self._run_recording_remux(
                            command,
                            origin,
                            session,
                        )
            except MediaSessionAdmissionError as exc:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if origin == 'prewarm':
                    raise RecordingPrewarmCancelled from exc
                raise HTTPException(
                    status_code=429,
                    detail=str(exc),
                    headers={'Retry-After': '1'},
                ) from exc
            except RecordingPrewarmCancelled:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise
            except subprocess.TimeoutExpired as exc:
                self._recording_cache_metric(origin, 'failures')
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=504, detail='recording fragment remux timed out') from exc
            except OSError as exc:
                self._recording_cache_metric(origin, 'failures')
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=500, detail=f'recording fragment remux failed: {exc}') from exc
            generated_init = temp_dir / 'init.mp4'
            generated_media = temp_dir / 'media_0.m4s'
            if result.returncode != 0 or not generated_init.exists() or (not generated_media.exists()):
                self._recording_cache_metric(origin, 'failures')
                error = (result.stderr or b'').decode('utf-8', errors='replace').strip()
                shutil.rmtree(temp_dir, ignore_errors=True)
                if time.time() - stat.st_mtime >= float(selected_config.recording_segment_seconds) * 2:
                    selected_manager.recorder.schedule_revalidation(path, error or 'recording fragment failed')
                with self.recording_day_cache_lock:
                    self.recording_day_cache.clear()
                raise HTTPException(status_code=500, detail=f'recording fragment failed: {error[-300:]}')
            try:
                offset_fmp4_timestamps(generated_init, generated_media, media_offset)
            except Exception as exc:
                self._recording_cache_metric(origin, 'failures')
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=500, detail=f'recording fragment timestamp repair failed: {exc}') from exc
            try:
                os.replace(generated_init, init_path)
                os.replace(generated_media, media_path)
            except OSError as exc:
                self._recording_cache_metric(origin, 'failures')
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=500, detail=f'recording fragment cache write failed: {exc}') from exc
            remux_ms = (time.monotonic() - remux_started) * 1000
            self._recording_cache_metric(origin, 'remuxes')
            self._recording_cache_metric(origin, 'remux_ms', remux_ms)
            with self.recording_cache_metrics_lock:
                self.recording_cache_metrics[f'{origin}_last_remux_ms'] = remux_ms
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._maintain_recording_cache(cache_dir)
            return (init_path, media_path)

    def _recording_cache_files_ready(self, init_path: Path, media_path: Path, *, touch: bool=False) -> bool:
        try:
            ready = init_path.is_file() and init_path.stat().st_size > 0 and media_path.is_file() and (media_path.stat().st_size > 0)
            if ready and touch:
                now = time.time()
                os.utime(init_path, (now, now))
                os.utime(media_path, (now, now))
            return ready
        except OSError:
            return False

    def _recording_file_response(self, path: Path, media_type: str) -> FileResponse:
        try:
            now = time.time()
            os.utime(path, (now, now))
        except OSError:
            raise HTTPException(status_code=404, detail='recording fragment cache entry disappeared')
        return FileResponse(path, media_type=media_type, headers={'Cache-Control': 'private, max-age=86400'})

    def _recording_preview_path(
        self,
        row: dict,
        epoch: float,
        *,
        width: int = 480,
        exact: bool = False,
        native_resolution: bool = False,
        active_manager: AppManager | None = None,
    ) -> Path:
        """Return a cached JPEG, optionally at native training resolution."""
        selected_manager = active_manager or self.manager
        selected_config = getattr(selected_manager, 'config', self.config)
        source_path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        start_epoch = float(row.get('start_epoch') or 0)
        end_epoch = float(row.get('end_epoch') or start_epoch)
        if not start_epoch <= epoch < end_epoch:
            raise HTTPException(status_code=404, detail='no recording exists at this time')
        duration = max(0.05, end_epoch - start_epoch)
        raw_offset = max(0.0, epoch - start_epoch)
        requested_width = 0 if native_resolution else max(320, min(1920, int(width)))
        preview_offset = min(
            max(
                0.0,
                raw_offset
                if exact
                else math.floor(raw_offset / self.recording_preview_interval_seconds)
                * self.recording_preview_interval_seconds,
            ),
            max(0.0, duration - 0.05),
        )
        try:
            stat = source_path.stat()
        except OSError as exc:
            raise HTTPException(status_code=404, detail='recording file not found') from exc
        fingerprint = (
            f'v3:{source_path}:{stat.st_mtime_ns}:{stat.st_size}:'
            f'{preview_offset:.3f}:{requested_width}:{int(exact)}'
        )
        cache_key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:32]
        cache_dir = selected_manager.database_dir / 'recording-preview-cache'
        preview_path = cache_dir / f'{cache_key}.jpg'
        metadata_path = preview_path.with_suffix('.json')
        if self._recording_preview_ready(preview_path, touch=True) and (
            not exact or metadata_path.is_file()
        ):
            return preview_path
        with self.recording_preview_locks_guard:
            lock = self.recording_preview_locks.setdefault(cache_key, threading.Lock())
        with lock:
            if self._recording_preview_ready(preview_path, touch=True) and (
                not exact or metadata_path.is_file()
            ):
                return preview_path
            if not self.recording_preview_build_limiter.acquire(timeout=3.0):
                raise HTTPException(status_code=429, detail='recording preview generator is busy', headers={'Retry-After': '1'})
            temporary = cache_dir / f'.{cache_key}.{os.getpid()}.{threading.get_ident()}.tmp.jpg'
            try:
                session = None
                media_sessions = getattr(selected_manager, 'media_sessions', None)
                if media_sessions is not None:
                    try:
                        session = media_sessions.acquire(
                            MediaSessionRequest(
                                MediaSessionKind.RECORDING_PREVIEW,
                                resources={MediaResourceClass.TRANSCODE_PROCESS: 1},
                                owner_generation=getattr(
                                    selected_manager,
                                    'media_session_generation',
                                    None,
                                ),
                            ),
                            timeout=3.0,
                        )
                    except MediaSessionAdmissionError as exc:
                        raise HTTPException(
                            status_code=429,
                            detail=str(exc),
                            headers={'Retry-After': '1'},
                        ) from exc
                cache_dir.mkdir(parents=True, exist_ok=True)
                jpeg_quality = 3 if native_resolution or requested_width > 480 else 5
                filters = ['showinfo@preview'] if exact else []
                if not native_resolution:
                    filters.append(f"scale='min({requested_width},iw)':-2")
                command = [
                    selected_config.ffmpeg_path, '-hide_banner',
                    '-loglevel', 'info' if exact else 'error',
                    *RECORDED_FRAME_INPUT_THREAD_ARGS,
                    '-ss', f'{preview_offset:.3f}', '-i', str(source_path),
                    '-map', '0:v:0', '-frames:v', '1',
                    *RECORDED_FRAME_OUTPUT_THREAD_ARGS,
                    *(['-vf', ','.join(filters)] if filters else []),
                    '-q:v', str(jpeg_quality), '-y', str(temporary),
                ]
                try:
                    if session is None:
                        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=8)
                    else:
                        with session:
                            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=8)
                except subprocess.TimeoutExpired as exc:
                    raise HTTPException(status_code=504, detail='recording preview timed out') from exc
                except OSError as exc:
                    raise HTTPException(status_code=500, detail='recording preview failed') from exc
                if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
                    error = (result.stderr or b'').decode('utf-8', errors='replace').strip()
                    LOGGER.warning('recording preview failed for %s at %.3f: %s', source_path.name, preview_offset, redact_secret_text(error[-300:]))
                    raise HTTPException(status_code=500, detail='recording preview failed')
                os.replace(temporary, preview_path)
                if exact:
                    timestamp_match = re.search(
                        rb'showinfo@preview[^\r\n]*\bpts_time:([-+0-9.eE]+)',
                        result.stderr or b'',
                    )
                    actual_epoch = None
                    timestamp_source = 'requested_offset'
                    if timestamp_match is not None:
                        actual_epoch = start_epoch + preview_offset + float(
                            timestamp_match.group(1)
                        )
                        timestamp_source = 'source_pts'
                    metadata_path.write_text(
                        json.dumps({
                            'actual_epoch': actual_epoch,
                            'requested_epoch': epoch,
                            'timestamp_source': timestamp_source,
                        }, separators=(',', ':')),
                        encoding='utf-8',
                    )
            finally:
                temporary.unlink(missing_ok=True)
                self.recording_preview_build_limiter.release()
            self._maintain_recording_preview_cache(preview_path)
            return preview_path

    def _recording_preview_strip(
        self,
        row: dict,
        *,
        width: int = 320,
        active_manager: AppManager | None = None,
    ) -> tuple[Path, dict]:
        """A cached JPEG grid of one segment's frames at a fixed step, and its layout.

        The timeline shows these while the playhead is dragged: one request
        covers a whole segment, so moving within it needs no further decode.
        Tile ``i`` is the frame ``i * step_seconds`` into the segment.
        """
        selected_manager = active_manager or self.manager
        selected_config = getattr(selected_manager, 'config', self.config)
        source_path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        start_epoch = float(row.get('start_epoch') or 0)
        end_epoch = float(row.get('end_epoch') or start_epoch)
        duration = max(0.05, end_epoch - start_epoch)
        tile_width = max(160, min(640, int(width)))
        # One frame a second; a long segment is thinned to at most 60 tiles.
        step = max(1, math.ceil(duration / RECORDING_STRIP_MAX_TILES))
        expected = math.ceil(duration / step) + 2
        columns = min(RECORDING_STRIP_COLUMNS, expected)
        rows = math.ceil(expected / columns)
        try:
            stat = source_path.stat()
        except OSError as exc:
            raise HTTPException(status_code=404, detail='recording file not found') from exc
        fingerprint = (
            f'strip-v1:{source_path}:{stat.st_mtime_ns}:{stat.st_size}:'
            f'{duration:.3f}:{tile_width}:{step}:{columns}x{rows}'
        )
        cache_key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:32]
        cache_dir = selected_manager.database_dir / 'recording-preview-cache'
        strip_path = cache_dir / f'{cache_key}.jpg'
        metadata_path = strip_path.with_suffix('.json')

        def cached() -> dict | None:
            if not self._recording_preview_ready(strip_path, touch=True):
                return None
            try:
                layout = json.loads(metadata_path.read_text(encoding='utf-8'))
                return layout if int(layout['count']) > 0 else None
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
                return None

        layout = cached()
        if layout is not None:
            return strip_path, layout
        with self.recording_preview_locks_guard:
            lock = self.recording_preview_locks.setdefault(cache_key, threading.Lock())
        with lock:
            layout = cached()
            if layout is not None:
                return strip_path, layout
            # A large frame costs most of its time in the decoder, so only
            # its keyframes are decoded: the frame for a second with no
            # keyframe is then the keyframe before it. Small streams (a
            # camera's sub stream) are cheap enough to decode in full.
            dimensions = mp4_video_dimensions(source_path)
            keyframes_only = (
                dimensions is None
                or dimensions[0] * dimensions[1] > RECORDING_STRIP_FULL_DECODE_PIXELS
            )
            # A large frame decodes and scales about twice as fast on the
            # GPU. It is tried where hardware is allowed and VAAPI works;
            # after a failure the CPU does the work for a while. Decided
            # before a build slot is taken: the first VAAPI probe is slow.
            hardware = (
                dimensions is not None
                and keyframes_only
                and self._hardware_acceleration_mode() != 'off'
                and time.monotonic() >= self.recording_strip_hardware_retry_at
                and bool(self._ffmpeg_vaapi_info().get('available'))
            )

            def command(on_gpu: bool) -> list[str]:
                # round=up: tile i is the last frame at or before i * step,
                # not one from the second that follows it. start_time=0: a
                # video track that begins a little into the file (audio came
                # first) would otherwise shift every tile by a step.
                select = f'fps=1/{step}:start_time=0:round=up'
                grid = f'showinfo@strip,tile={columns}x{rows}'
                if on_gpu:
                    device = str(self._ffmpeg_vaapi_info().get('device') or '/dev/dri/renderD128')
                    decode = [
                        '-vaapi_device', device,
                        '-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi',
                    ]
                    filters = (
                        f'{select},scale_vaapi=w={tile_width}:h=-2:format=nv12,'
                        f'hwdownload,format=nv12,{grid}'
                    )
                else:
                    decode = [*RECORDED_FRAME_INPUT_THREAD_ARGS]
                    filters = f"{select},scale='min({tile_width},iw)':-2,{grid}"
                return [
                    selected_config.ffmpeg_path, '-hide_banner', '-loglevel', 'info', '-nostdin',
                    *decode,
                    *(['-skip_frame', 'nokey'] if keyframes_only else []),
                    '-i', str(source_path),
                    '-map', '0:v:0', '-an', '-frames:v', '1',
                    *RECORDED_FRAME_OUTPUT_THREAD_ARGS,
                    '-vf', filters,
                    '-q:v', '6', '-y', str(temporary),
                ]

            def build(on_gpu: bool, timeout: float) -> tuple[int, str]:
                """(tiles that hold a frame, or 0 on failure; why it failed)."""
                temporary.unlink(missing_ok=True)
                built = subprocess.run(
                    command(on_gpu), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout,
                )
                # The grid is padded with blank tiles; only these hold a frame.
                frames = min(
                    columns * rows,
                    len(re.findall(rb'showinfo@strip[^\r\n]*\bpts_time:', built.stderr or b'')),
                )
                if (
                    built.returncode == 0 and frames > 0
                    and temporary.is_file() and temporary.stat().st_size > 0
                ):
                    return frames, ''
                return 0, (built.stderr or b'').decode('utf-8', errors='replace').strip()[-300:]

            temporary = cache_dir / f'.{cache_key}.{os.getpid()}.{threading.get_ident()}.tmp.jpg'
            if not self.recording_strip_build_limiter.acquire(timeout=3.0):
                raise HTTPException(status_code=429, detail='recording preview generator is busy', headers={'Retry-After': '1'})
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                session = None
                media_sessions = getattr(selected_manager, 'media_sessions', None)
                if media_sessions is not None:
                    try:
                        # A strip is a short decode that ends in one small
                        # JPEG. It has a resource of its own: two strips at
                        # once would take both transcode slots that event
                        # clips, exports and HomeKit share, and the JPEG
                        # encoders can all be held by live MJPEG streams.
                        session = media_sessions.acquire(
                            MediaSessionRequest(
                                MediaSessionKind.RECORDING_PREVIEW,
                                resources={MediaResourceClass.PREVIEW_PROCESS: 1},
                                owner_generation=getattr(
                                    selected_manager,
                                    'media_session_generation',
                                    None,
                                ),
                            ),
                            timeout=3.0,
                        )
                    except MediaSessionAdmissionError as exc:
                        raise HTTPException(
                            status_code=429,
                            detail=str(exc),
                            headers={'Retry-After': '1'},
                        ) from exc
                # Nothing between taking the session and entering it: a lease
                # that is never entered is never released.
                with session if session is not None else contextlib.nullcontext():
                    count, error = 0, ''
                    if hardware:
                        try:
                            count, error = build(True, RECORDING_STRIP_HARDWARE_TIMEOUT_SECONDS)
                        except subprocess.TimeoutExpired:
                            error = 'timed out'
                        except OSError as exc:
                            error = str(exc)
                        if count <= 0:
                            # A hung or broken GPU must not cost every strip
                            # its timeout: the CPU takes over for a while.
                            self.recording_strip_hardware_retry_at = (
                                time.monotonic() + RECORDING_STRIP_HARDWARE_RETRY_SECONDS
                            )
                            LOGGER.warning(
                                'recording preview strip: hardware decode failed for %s, using the CPU: %s',
                                source_path.name, redact_secret_text(error),
                            )
                            hardware = False
                    if count <= 0:
                        try:
                            count, error = build(False, 15)
                        except subprocess.TimeoutExpired as exc:
                            raise HTTPException(status_code=504, detail='recording preview timed out') from exc
                        except OSError as exc:
                            raise HTTPException(status_code=500, detail='recording preview failed') from exc
                if count <= 0:
                    LOGGER.warning('recording preview strip failed for %s: %s', source_path.name, redact_secret_text(error))
                    raise HTTPException(status_code=500, detail='recording preview failed')
                layout = {
                    'count': count,
                    'columns': columns,
                    'rows': rows,
                    'step_seconds': step,
                    'keyframes_only': keyframes_only,
                    'hardware': hardware,
                }
                # The layout first: an image without one is never served.
                metadata_path.write_text(json.dumps(layout, separators=(',', ':')), encoding='utf-8')
                os.replace(temporary, strip_path)
            finally:
                temporary.unlink(missing_ok=True)
                self.recording_strip_build_limiter.release()
            self._maintain_recording_preview_cache(strip_path)
            return strip_path, layout

    def _recording_segment_frames(
        self,
        row: dict,
        *,
        active_manager: AppManager | None = None,
    ) -> SegmentFrames:
        """The frames one recording file presents, from its packet index.

        A demux, not a decode (tens of milliseconds for a ten-second
        segment); remembered per file version, for a bounded number of files.
        """
        selected_manager = active_manager or self.manager
        source_path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        try:
            stat = source_path.stat()
        except OSError as exc:
            raise HTTPException(status_code=404, detail='recording file not found') from exc
        key = (str(source_path), stat.st_mtime_ns, stat.st_size)
        with self.recording_frame_list_lock:
            cached = self.recording_frame_lists.get(key)
            if cached is not None:
                self.recording_frame_lists.move_to_end(key)
                return cached
        command = [
            self.deps.ffprobe_path(), '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'packet=pts_time,flags:format=start_time',
            '-of', 'json', str(source_path),
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail='recording frame list timed out') from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail='recording frame list failed') from exc
        frames = parse_packet_listing((result.stdout or b'').decode('utf-8', errors='replace'))
        if result.returncode != 0 or not frames.times:
            error = (result.stderr or b'').decode('utf-8', errors='replace').strip()
            LOGGER.warning(
                'recording frame list failed for %s: %s',
                source_path.name, redact_secret_text(error[-300:]) or 'no video frames',
            )
            raise HTTPException(status_code=500, detail='recording frame list failed')
        with self.recording_frame_list_lock:
            self.recording_frame_lists[key] = frames
            self.recording_frame_lists.move_to_end(key)
            while len(self.recording_frame_lists) > RECORDING_FRAME_LIST_CACHE_ENTRIES:
                self.recording_frame_lists.popitem(last=False)
        return frames

    def _recording_step_frame(
        self,
        row: dict,
        epoch: float,
        *,
        width: int = 1280,
        active_manager: AppManager | None = None,
    ) -> tuple[Path, float]:
        """The exact frame on screen at ``epoch``, as a cached JPEG, and its time.

        The frame is decoded with the rest of its run (``frame_run``) in one
        decoder process; the whole run is cached, so stepping on through it
        reads files. Returns the JPEG and the epoch of the frame it shows,
        from the decoded frame's own timestamp.
        """
        selected_manager = active_manager or self.manager
        selected_config = getattr(selected_manager, 'config', self.config)
        source_path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        start_epoch = float(row.get('start_epoch') or 0)
        frames = self._recording_segment_frames(row, active_manager=selected_manager)
        index = frame_index_at(frames.times, epoch - start_epoch)
        run_start, run_stop = frame_run(frames, index)
        target = frames.times[index]
        output_width = max(320, min(1920, int(width)))
        try:
            stat = source_path.stat()
        except OSError as exc:
            raise HTTPException(status_code=404, detail='recording file not found') from exc
        fingerprint = (
            f'frame-run-v1:{source_path}:{stat.st_mtime_ns}:{stat.st_size}:'
            f'{frames.times[run_start]:.6f}:{run_stop - run_start}:{output_width}'
        )
        cache_key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:32]
        cache_dir = selected_manager.database_dir / 'recording-preview-cache'
        # Every file is a JPEG with an optional sibling .json, so the preview
        # cache's size and age bounds cover runs too. The run's frame times
        # sit beside its first frame: losing that frame loses the run.
        metadata_path = cache_dir / f'{cache_key}-00.json'

        def frame_path(number: int) -> Path:
            return cache_dir / f'{cache_key}-{number:02d}.jpg'

        def cached() -> tuple[Path, float] | None:
            try:
                times = [float(value) for value in json.loads(metadata_path.read_text(encoding='utf-8'))['times']]
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
                return None
            if not times:
                return None
            number = frame_index_at(times, target)
            path = frame_path(number)
            if not self._recording_preview_ready(path, touch=True):
                return None
            return path, start_epoch + times[number]

        found = cached()
        if found is not None:
            return found
        with self.recording_preview_locks_guard:
            lock = self.recording_preview_locks.setdefault(cache_key, threading.Lock())
        with lock:
            found = cached()
            if found is not None:
                return found
            if not self.recording_frame_run_build_limiter.acquire(timeout=3.0):
                raise HTTPException(status_code=429, detail='recording frame generator is busy', headers={'Retry-After': '1'})
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                session = None
                media_sessions = getattr(selected_manager, 'media_sessions', None)
                if media_sessions is not None:
                    try:
                        # Like a strip: a short decode ending in a few
                        # JPEGs, never one of the two transcode slots event
                        # clips, exports and HomeKit share.
                        session = media_sessions.acquire(
                            MediaSessionRequest(
                                MediaSessionKind.RECORDING_PREVIEW,
                                resources={MediaResourceClass.PREVIEW_PROCESS: 1},
                                owner_generation=getattr(
                                    selected_manager,
                                    'media_session_generation',
                                    None,
                                ),
                            ),
                            timeout=3.0,
                        )
                    except MediaSessionAdmissionError as exc:
                        raise HTTPException(
                            status_code=429,
                            detail=str(exc),
                            headers={'Retry-After': '1'},
                        ) from exc
                # Nothing between taking the session and entering it.
                with session if session is not None else contextlib.nullcontext():
                    decoded = self._decode_frame_run(
                        selected_config.ffmpeg_path, source_path, frames, run_start, run_stop, output_width,
                    )
                times = []
                for number, (pts, image) in enumerate(decoded):
                    temporary = cache_dir / f'.{cache_key}-{number:02d}.{os.getpid()}.{threading.get_ident()}.tmp.jpg'
                    try:
                        temporary.write_bytes(image)
                        os.replace(temporary, frame_path(number))
                    finally:
                        temporary.unlink(missing_ok=True)
                    times.append(round(pts - frames.start_time, 6))
                # The times last: a run without them is never served.
                metadata_path.write_text(json.dumps({'times': times}, separators=(',', ':')), encoding='utf-8')
            finally:
                self.recording_frame_run_build_limiter.release()
            self._maintain_recording_preview_cache(frame_path(0))
            found = cached()
            if found is None:
                raise HTTPException(status_code=500, detail='recording frame failed')
            return found

    def _decode_frame_run(
        self,
        ffmpeg_path: str,
        source_path: Path,
        frames: SegmentFrames,
        run_start: int,
        run_stop: int,
        width: int,
    ) -> list[tuple[float, bytes]]:
        """Decode frames ``[run_start, run_stop)`` once: (source pts, JPEG) each.

        Goes through the shared recorded-frame decoder: its thread limits,
        hardware plans (GPU scale where VAAPI or QSV is configured, the CPU
        otherwise or after a GPU failure) and failure classes.
        """
        size = scaled_size(mp4_video_dimensions(source_path), width)
        plans = recorded_decode_plans(self._hardware_acceleration_mode(), scale=size) if size else (DecodePlan('cpu'),)
        count = run_stop - run_start
        # Accurate seek: decoding starts at the keyframe before this time and
        # frames before it are dropped. Copied timestamps keep the source PTS.
        seek = f'{max(0.0, frames.start_time + frames.times[run_start] - FRAME_TIME_TOLERANCE_SECONDS):.6f}'
        result = None
        for plan in plans:
            if plan.backend == 'cpu':
                scale = f'scale={size[0]}:{size[1]}' if size else f"scale='min({width},iw)':-2"
                video_filter = f'{scale},showinfo'
            else:
                video_filter = f'{plan.download_filter},showinfo'
            command = build_decode_command(
                ffmpeg_path, source_path, plan=plan, loglevel='info', seek=seek, copyts=True,
                first_video_stream=True, video_filter=video_filter,
                passthrough_timestamps=True, frames=count,
            )
            result = run_decoder(
                command, backend=plan.backend, timeout=RECORDING_FRAME_RUN_TIMEOUT_SECONDS,
                stdout_reader=read_bmp_frames,
            )
            warn_decoder_errors('frame_step', source_path, plan.backend, result.stderr)
            if result.ok and result.output:
                break
            warn_decoder_failure('frame_step', source_path, result)
        if result is None or not result.ok or not result.output:
            if result is not None and result.failure == FAILURE_TIMEOUT:
                raise HTTPException(status_code=504, detail='recording frame timed out')
            raise HTTPException(status_code=500, detail='recording frame failed')
        decoded: list[tuple[float, bytes]] = []
        for pts, image in zip(showinfo_times(result.stderr), result.output):
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, RECORDING_FRAME_JPEG_QUALITY])
            if not ok:
                raise HTTPException(status_code=500, detail='recording frame failed')
            decoded.append((pts, encoded.tobytes()))
        if not decoded:
            # Frames without their times cannot be placed; nothing is cached.
            LOGGER.warning('recording frame run for %s reported no frame times', source_path.name)
            raise HTTPException(status_code=500, detail='recording frame failed')
        return decoded

    @staticmethod
    def _recording_preview_timestamp(path: Path) -> tuple[float | None, str]:
        """Return persisted source timing for a generated recording preview."""
        metadata_path = path.with_suffix('.json')
        try:
            payload = json.loads(metadata_path.read_text(encoding='utf-8'))
            actual = payload.get('actual_epoch')
            source = str(payload.get('timestamp_source') or 'requested_offset')
            return (float(actual) if actual is not None else None, source)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None, 'requested_offset'

    def _recording_preview_ready(self, path: Path, *, touch: bool=False) -> bool:
        try:
            ready = path.is_file() and path.stat().st_size > 0
            if ready and touch:
                now = time.time()
                os.utime(path, (now, now))
            return ready
        except OSError:
            return False

    def _maintain_recording_preview_cache(self, active_path: Path) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self.recording_preview_last_maintenance < 600:
            return
        if not self.recording_preview_maintenance_lock.acquire(blocking=False):
            return
        try:
            self.recording_preview_last_maintenance = now_monotonic
            cache_dir = active_path.parent
            now_epoch = time.time()
            entries: list[tuple[float, int, Path]] = []
            for path in cache_dir.glob('*.jpg'):
                if path == active_path:
                    continue
                try:
                    stat = path.stat()
                    if now_epoch - stat.st_mtime > self.recording_preview_max_age_seconds:
                        path.unlink(missing_ok=True)
                        path.with_suffix('.json').unlink(missing_ok=True)
                    else:
                        entries.append((stat.st_mtime, stat.st_size, path))
                except OSError:
                    continue
            try:
                active_size = active_path.stat().st_size
            except OSError:
                active_size = 0
            total_size = sum((size for _, size, _ in entries)) + active_size
            for _modified_at, size, path in sorted(entries):
                if total_size <= self.recording_preview_max_bytes:
                    break
                try:
                    path.unlink(missing_ok=True)
                    path.with_suffix('.json').unlink(missing_ok=True)
                    total_size -= size
                except OSError:
                    continue
        finally:
            self.recording_preview_maintenance_lock.release()

    def _maintain_recording_cache(self, active_dir: Path) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self.recording_cache_last_maintenance < 600:
            return
        if not self.recording_cache_maintenance_lock.acquire(blocking=False):
            return
        try:
            self.recording_cache_last_maintenance = now_monotonic
            root = self.manager.storage_dir / 'playback-cache' / 'fmp4'
            if not root.exists():
                return
            now_epoch = time.time()
            entries: list[tuple[float, int, Path]] = []
            for directory in root.iterdir():
                if not directory.is_dir() or directory == active_dir:
                    continue
                with self.recording_fmp4_locks_guard:
                    entry_lock = self.recording_fmp4_locks.setdefault(directory.name, threading.Lock())
                if not entry_lock.acquire(blocking=False):
                    continue
                try:
                    for child in directory.iterdir():
                        if child.is_dir() and child.name.startswith('fmp4-'):
                            try:
                                if now_epoch - child.stat().st_mtime > 300:
                                    shutil.rmtree(child, ignore_errors=True)
                            except OSError:
                                continue
                    files = [item for item in directory.iterdir() if item.is_file()]
                    modified_at = max((item.stat().st_mtime for item in files), default=directory.stat().st_mtime)
                    size = sum((item.stat().st_size for item in files))
                    max_age_seconds = int(self.config.recording_cache_max_days) * 24 * 60 * 60
                    if now_epoch - modified_at > max_age_seconds:
                        shutil.rmtree(directory, ignore_errors=True)
                    else:
                        entries.append((modified_at, size, directory))
                except OSError:
                    continue
                finally:
                    entry_lock.release()
            total_size = sum((size for _, size, _ in entries))
            max_bytes = int(float(self.config.recording_cache_max_gb) * 1024 * 1024 * 1024)
            for modified_at, size, directory in sorted(entries):
                if total_size <= max_bytes:
                    break
                if now_epoch - modified_at < 300:
                    continue
                with self.recording_fmp4_locks_guard:
                    entry_lock = self.recording_fmp4_locks.setdefault(directory.name, threading.Lock())
                if not entry_lock.acquire(blocking=False):
                    continue
                try:
                    shutil.rmtree(directory, ignore_errors=True)
                    total_size -= size
                finally:
                    entry_lock.release()
        finally:
            self.recording_cache_maintenance_lock.release()

    def _start_recording_prewarmer(self) -> None:
        if self.recording_prewarm_thread is not None and self.recording_prewarm_thread.is_alive():
            return
        self.recording_prewarm_stop.clear()
        thread = threading.Thread(target=self._recording_prewarm_loop, name='recording-prewarmer', daemon=False)
        self.recording_prewarm_thread = thread
        try:
            thread.start()
        except BaseException:
            self.recording_prewarm_thread = None
            self.recording_prewarm_stop.set()
            raise

    def _stop_recording_prewarmer(self) -> None:
        logger = logging.getLogger(__name__)
        self.recording_prewarm_stop.set()
        with self.recording_prewarm_process_lock:
            process = self.recording_prewarm_process
        if process is not None:
            self._signal_recording_prewarm_process(process, signal.SIGTERM)
        if self.recording_prewarm_thread is not None:
            self.recording_prewarm_thread.join(timeout=5)
            if self.recording_prewarm_thread.is_alive():
                with self.recording_prewarm_process_lock:
                    process = self.recording_prewarm_process
                if process is not None:
                    self._signal_recording_prewarm_process(process, signal.SIGKILL)
                self.recording_prewarm_thread.join(timeout=3)
            if self.recording_prewarm_thread.is_alive():
                logger.error('recording prewarmer did not stop after cancellation')
                raise RuntimeError('recording prewarmer did not stop after cancellation')
            else:
                self.recording_prewarm_thread = None

    def _recording_prewarm_loop(self) -> None:
        while not self.recording_prewarm_stop.wait(5):
            if not self.config.recording_cache_prewarm:
                continue
            for camera in self.config.cameras:
                sources = []
                if camera.record:
                    sources.append('main')
                if camera.record_sub and camera.live_stream_url:
                    sources.append('live')
                for source in sources:
                    if self.recording_prewarm_stop.is_set():
                        return
                    try:
                        row = self.manager.recorder.latest_indexed_row(camera.id, source)
                        if row is None:
                            continue
                        path = Path(row['path'])
                        if not path.exists() or time.time() - path.stat().st_mtime < float(self.config.recording_segment_seconds) * 2:
                            continue
                        window_start, window_end = self.deps.recording_playback_window(float(row['start_epoch']))
                        rows = self.manager.recorder.recording_rows_between(camera.id, window_start, window_end, source, discover_missing=False)
                        targets = [rows[0], row] if rows else []
                        warmed: set[str] = set()
                        for target in targets:
                            target_path = str(target['path'])
                            if target_path in warmed:
                                continue
                            index = next((i for i, item in enumerate(rows) if item['path'] == target_path), None)
                            if index is None:
                                continue
                            warmed.add(target_path)
                            media_offset = sum((float(item['duration_seconds']) for item in rows[:index]))
                            self._recording_fmp4_files(Path(target_path), float(target['duration_seconds']), media_offset, origin='prewarm')
                    except RecordingPrewarmCancelled:
                        return
                    except Exception:
                        logging.getLogger(__name__).exception('Recording prewarm failed for %s/%s', camera.id, source)

    def _recording_day_fmp4_paths(self, camera_id: str, segment_name: str, start_epoch: float, end_epoch: float, source: str='main', media_offset: float=0.0, trim_end: bool=False, *, active_manager: AppManager | None=None) -> tuple[Path, Path]:
        selected_manager = active_manager or self.manager
        if selected_manager.camera(camera_id) is None:
            raise HTTPException(status_code=404, detail='camera not found')
        self.deps.validate_recording_range(start_epoch, end_epoch, 90000, 'invalid recording day range')
        if not math.isfinite(media_offset) or media_offset < 0:
            raise HTTPException(status_code=400, detail='invalid recording media offset')
        rows = self._recording_day_rows(camera_id, start_epoch, end_epoch, source, active_manager=selected_manager)
        if not segment_name or Path(segment_name).name != segment_name:
            raise HTTPException(status_code=404, detail='recording segment not found')
        segment_index = next((index for index, row in enumerate(rows) if str(row.get('name') or '') == segment_name), None)
        if segment_index is None:
            raise HTTPException(status_code=404, detail='recording segment not found')
        row = rows[segment_index]
        path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        segment_duration = playback_segment_duration(float(row['start_epoch']), float(row['duration_seconds']), end_epoch, trim_end)
        expected_offset = sum((float(row['duration_seconds']) for row in rows[:segment_index]))
        if abs(media_offset - expected_offset) > 0.1:
            media_offset = expected_offset
        return self._recording_fmp4_files(
            path,
            segment_duration,
            media_offset,
            active_manager=selected_manager,
        )

    def _recording_segment_path(self, camera_id: str, epoch: float, source: str='main', *, active_manager: AppManager | None=None) -> LeasedRecordingSegment:
        """Resolve the indexed source MP4 containing an epoch for native playback."""
        selected_manager = active_manager or self.manager
        if selected_manager.camera(camera_id) is None:
            raise HTTPException(status_code=404, detail='camera not found')
        if not math.isfinite(epoch) or epoch <= 0:
            raise HTTPException(status_code=400, detail='invalid recording segment time')
        rows = selected_manager.recorder.recording_rows_between(
            camera_id,
            epoch - 0.001,
            epoch + 0.001,
            recording_source(source),
            discover_missing=False,
        )
        row = next(
            (
                candidate for candidate in rows
                if float(candidate.get('start_epoch') or 0) <= epoch
                < float(candidate.get('end_epoch') or candidate.get('start_epoch') or 0)
            ),
            None,
        )
        if row is None:
            raise HTTPException(status_code=404, detail='no recording exists at this time')
        token = selected_manager.recorder.acquire_recording_for_playback(row)
        if token is None:
            raise HTTPException(status_code=404, detail='recording is being removed')
        try:
            path = self._recording_storage_path(
                row.get('path'),
                active_manager=selected_manager,
            )
            return LeasedRecordingSegment(path, token)
        except Exception:
            selected_manager.recorder.release_recording_playback(token)
            raise

    def _recording_rows(self, camera_id: str, limit: int, source: str='main', *, active_manager: AppManager | None=None) -> list[dict]:
        selected_manager = active_manager or self.manager
        return selected_manager.recorder.recording_rows(camera_id, limit=limit, source=recording_source(source))

    def _recording_storage_path(self, value: object, *, active_manager: AppManager | None=None) -> Path:
        selected_manager = active_manager or self.manager
        if not value:
            raise HTTPException(status_code=404, detail='recording file not found')
        try:
            path = Path(str(value)).resolve(strict=True)
            media_storage = getattr(selected_manager, "media_storage", None)
            if media_storage is not None:
                if not media_storage.contains(path, "recordings"):
                    raise ValueError("recording file is outside configured media storage")
            else:
                path.relative_to(selected_manager.recorder.recordings_dir.resolve())
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail='recording file not found') from None
        except (OSError, ValueError):
            raise HTTPException(status_code=403, detail='recording file is outside storage') from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail='recording file not found')
        return path

    def _event_clip_window(self, before: float | None, after: float | None, *, active_manager: AppManager | None=None) -> tuple[float, float]:
        selected_config = (active_manager or self.manager).config
        try:
            return event_clip_window(selected_config.event_clip_before_seconds, selected_config.event_clip_after_seconds, before, after)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    def _event_clip_path(self, event: dict, before: float, after: float, source: str='main', *, active_manager: AppManager | None=None) -> Path:
        selected_manager = active_manager or self.manager
        event_id = int(event.get('id') or 0)
        camera_id = slugify_camera_id(str(event.get('camera_id') or 'camera'))
        safe_before = int(max(0.0, min(float(before), 3600.0)) * 1000)
        safe_after = int(max(0.0, min(float(after), 3600.0)) * 1000)
        clip_source = recording_source(source)
        media_storage = getattr(selected_manager, "media_storage", None)
        clip_dir = (
            media_storage.directory(
                'clips', f'{camera_id}:{clip_source}', camera_id, clip_source
            )
            if media_storage is not None
            else selected_manager.storage_dir / 'event_clips' / camera_id / clip_source
        )
        clip_dir.mkdir(parents=True, exist_ok=True)
        accel_mode = self._hardware_acceleration_mode()
        anchor = event.get('scene_clip_start_epoch')
        suffix = f'-scene-{int(float(anchor) * 1000)}' if anchor is not None else ''
        return clip_dir / f'{event_id}-{safe_before}-{safe_after}-a3-{accel_mode}{suffix}.mp4'

    def _ensure_event_clip(self, event: dict, *, before: float, after: float, source: str='main', active_manager: AppManager | None=None) -> Path:
        selected_manager = active_manager or self.manager
        clip_source = recording_source(source)
        # Freeze the same input rows for the cache identity and the builder.
        # Newly finalized segments must create a new artifact for a growing window.
        window_start = event_epoch(event) - before
        window_end = event_epoch(event) + after
        rows = event.get("_recording_rows")
        if rows is None:
            rows = selected_manager.recorder.recording_rows_between(
                str(event.get("camera_id") or ""), window_start, window_end,
                clip_source, discover_missing=False,
            )
        rows = sorted(
            (dict(row) for row in rows if row.get("start_epoch") is not None
             and row.get("end_epoch") is not None
             and float(row["end_epoch"]) > window_start
             and float(row["start_epoch"]) < window_end),
            key=lambda row: (float(row["start_epoch"]), str(row.get("path") or "")),
        )
        available = []
        for row in rows:
            try:
                path = self._recording_storage_path(row.get("path"), active_manager=selected_manager)
            except HTTPException:
                continue
            available.append({**row, "path": str(path)})
        rows = available
        event = {**event, "_recording_rows": rows}
        revision = hashlib.sha256(json.dumps([
            [row.get(key) for key in ("path", "size_bytes", "modified_at", "start_epoch", "end_epoch", "duration_seconds")]
            for row in rows
        ], separators=(",", ":")).encode()).hexdigest()[:24]
        base_path = self._event_clip_path(event, before=before, after=after, source=clip_source, active_manager=selected_manager)
        clip_path = base_path.with_name(f"{base_path.stem}-r{revision}{base_path.suffix}")
        if clip_path.exists() and clip_path.stat().st_size > 0:
            return clip_path
        cache_key = str(clip_path)
        with self.event_clip_locks_guard:
            lock = self.event_clip_locks.setdefault(cache_key, threading.Lock())
        with lock:
            if clip_path.exists() and clip_path.stat().st_size > 0:
                return clip_path
            if not self.event_clip_build_limiter.acquire(blocking=False):
                raise HTTPException(status_code=429, detail='too many event clips are already being generated', headers={'Retry-After': '3'})
            try:
                session = None
                media_sessions = getattr(selected_manager, 'media_sessions', None)
                if media_sessions is not None:
                    try:
                        session = media_sessions.acquire(
                            MediaSessionRequest(
                                MediaSessionKind.EVENT_CLIP,
                                camera_id=str(event.get('camera_id') or '') or None,
                                source=clip_source,
                                resources={MediaResourceClass.TRANSCODE_PROCESS: 1},
                                owner_generation=getattr(
                                    selected_manager,
                                    'media_session_generation',
                                    None,
                                ),
                            ),
                            blocking=False,
                        )
                    except MediaSessionAdmissionError as exc:
                        raise HTTPException(
                            status_code=429,
                            detail=str(exc),
                            headers={'Retry-After': '3'},
                        ) from exc
                if session is None:
                    self._build_event_clip(event, before=before, after=after, output_path=clip_path, source=clip_source, active_manager=selected_manager)
                else:
                    with session:
                        self._build_event_clip(event, before=before, after=after, output_path=clip_path, source=clip_source, active_manager=selected_manager)
            finally:
                self.event_clip_build_limiter.release()
        return clip_path

    def _build_event_clip(self, event: dict, before: float, after: float, output_path: Path, source: str='main', *, active_manager: AppManager | None=None) -> None:
        selected_manager = active_manager or self.manager
        camera_id = str(event.get('camera_id') or '')
        if not camera_id:
            raise HTTPException(status_code=400, detail='event is missing camera')
        event_created_epoch = event_epoch(event)
        window_before = max(0.0, min(float(before), 3600.0))
        window_after = max(0.0, min(float(after), 3600.0))
        window_start = event_created_epoch - window_before
        window_end = event_created_epoch + window_after
        rows: list[dict] = []
        candidates = event.get("_recording_rows")
        if candidates is None:
            candidates = selected_manager.recorder.recording_rows_between(camera_id, window_start, window_end, recording_source(source), discover_missing=False)
        for candidate in candidates:
            if candidate.get('start_epoch') is None or candidate.get('end_epoch') is None:
                continue
            try:
                candidate = {**candidate, 'path': str(self._recording_storage_path(candidate.get('path'), active_manager=selected_manager))}
            except HTTPException:
                if "_recording_rows" in event:
                    # Never publish partial bytes under a frozen input revision.
                    raise
                continue
            rows.append(candidate)
        rows.sort(key=lambda row: float(row['start_epoch']))
        selected = [row for row in rows if float(row['end_epoch']) > window_start and float(row['start_epoch']) < window_end]
        if not selected:
            raise HTTPException(status_code=404, detail='no recording window found')
        try:
            local_start, duration = concatenated_clip_timing(selected, window_start, window_end)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        concat_path = self._write_concat_file(selected, active_manager=selected_manager)
        tmp_path = output_path.with_name(f'.{output_path.stem}.{os.getpid()}.tmp.mp4')
        source_codec = self._probe_video_codec(Path(str(selected[0]['path'])))
        commands: list[tuple[str, list[str]]] = []
        if self._event_clip_vaapi_enabled(source_codec):
            commands.append(('vaapi', self._event_clip_vaapi_command(source_codec, concat_path, local_start, duration, tmp_path)))
        if self._event_clip_qsv_enabled(source_codec):
            commands.append(('qsv', self._event_clip_qsv_command(source_codec, concat_path, local_start, duration, tmp_path)))
        commands.append(('cpu', self._event_clip_cpu_command(concat_path, local_start, duration, tmp_path)))
        try:
            last_error = 'event clip generation failed'
            for backend, command in commands:
                tmp_path.unlink(missing_ok=True)
                clip_timeout = max(60.0, min(600.0, duration * 2.0))
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=clip_timeout)
                if result.returncode == 0 and tmp_path.exists() and (tmp_path.stat().st_size > 0):
                    tmp_path.replace(output_path)
                    logging.getLogger(__name__).info('built event clip %s using %s acceleration (source codec %s)', output_path.name, backend, source_codec or 'unknown')
                    return
                last_error = (result.stderr or f'event clip generation failed using {backend}').strip()[-500:]
                if backend in {'vaapi', 'qsv'}:
                    logging.getLogger(__name__).warning('%s event clip generation failed for %s, falling back to next backend: %s', backend.upper(), output_path.name, last_error)
            logging.getLogger(__name__).error('event clip generation failed for %s: %s', output_path.name, redact_secret_text(last_error))
            raise HTTPException(status_code=500, detail='event clip generation failed')
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail='event clip generation timed out') from exc
        finally:
            concat_path.unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)

    def _write_concat_file(self, rows: list[dict], *, active_manager: AppManager | None=None) -> Path:
        selected_manager = active_manager or self.manager
        paths = [str(self._recording_storage_path(row.get('path'), active_manager=selected_manager)) for row in rows]
        if any(('\n' in path or '\r' in path for path in paths)):
            raise HTTPException(status_code=400, detail='recording path is invalid')
        handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.ffconcat', prefix='survng-recordings-', delete=False)
        with handle:
            for path_value in paths:
                escaped = path_value.replace('\\', '\\\\').replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        return Path(handle.name)
