"""Recording media runtime ownership: caching, remux, previews, clips, exports, and hardware probes."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import mmap
import os
import re
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

from .config import AppConfig, slugify_camera_id
from .incident_utils import event_epoch
from .manager import AppManager
from .media_exports import MediaExportManager
from .recording_media import concatenated_clip_timing, event_clip_window, playback_segment_duration
from .recording_routes import recording_source
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)


class RecordingPrewarmCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecordingMediaDependencies:
    get_config: Callable[[], AppConfig]
    get_manager: Callable[[], AppManager]
    ffprobe_path: Callable[[], str]
    validate_recording_range: Callable[[float, float, float, str], None]
    recording_playback_window: Callable[[float], tuple[float, float]]


class RecordingMediaRuntime:
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
        self.recording_preview_interval_seconds = 5.0
        self.recording_preview_max_age_seconds = 7 * 24 * 60 * 60
        self.recording_preview_max_bytes = 256 * 1024 * 1024
        self.recording_preview_build_limiter = threading.BoundedSemaphore(1)
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

    def _mp4_boxes(self, data: bytes | bytearray, start: int=0, end: int | None=None):
        limit = len(data) if end is None else min(end, len(data))
        cursor = start
        while cursor + 8 <= limit:
            size = struct.unpack_from('>I', data, cursor)[0]
            box_type = bytes(data[cursor + 4:cursor + 8])
            header = 8
            if size == 1 and cursor + 16 <= limit:
                size = struct.unpack_from('>Q', data, cursor + 8)[0]
                header = 16
            elif size == 0:
                size = limit - cursor
            if size < header or cursor + size > limit:
                break
            yield (box_type, cursor, cursor + header, cursor + size)
            cursor += size

    def _mp4_track_timescales(self, init_data: bytes) -> dict[int, int]:
        timescales: dict[int, int] = {}
        for box_type, _, payload, box_end in self._mp4_boxes(init_data):
            if box_type != b'moov':
                continue
            for child_type, _, child_payload, child_end in self._mp4_boxes(init_data, payload, box_end):
                if child_type != b'trak':
                    continue
                track_id = None
                timescale = None
                for trak_type, _, trak_payload, trak_end in self._mp4_boxes(init_data, child_payload, child_end):
                    if trak_type == b'tkhd':
                        version = init_data[trak_payload]
                        offset = trak_payload + (20 if version == 1 else 12)
                        if offset + 4 <= trak_end:
                            track_id = struct.unpack_from('>I', init_data, offset)[0]
                    elif trak_type == b'mdia':
                        for mdia_type, _, mdia_payload, mdia_end in self._mp4_boxes(init_data, trak_payload, trak_end):
                            if mdia_type != b'mdhd':
                                continue
                            version = init_data[mdia_payload]
                            offset = mdia_payload + (20 if version == 1 else 12)
                            if offset + 4 <= mdia_end:
                                timescale = struct.unpack_from('>I', init_data, offset)[0]
                if track_id and timescale:
                    timescales[track_id] = timescale
        return timescales

    def _offset_fmp4_timestamps(self, init_path: Path, media_path: Path, seconds: float) -> None:
        if seconds <= 0:
            return
        timescales = self._mp4_track_timescales(init_path.read_bytes())
        if not timescales:
            raise RuntimeError('fragment init has no track timescales')
        adjusted = 0
        with media_path.open('r+b') as media_file, mmap.mmap(media_file.fileno(), 0) as data:
            for box_type, _, payload, box_end in self._mp4_boxes(data):
                if box_type != b'moof':
                    continue
                for child_type, _, child_payload, child_end in self._mp4_boxes(data, payload, box_end):
                    if child_type != b'traf':
                        continue
                    track_id = None
                    tfdt = None
                    for traf_type, _, traf_payload, traf_end in self._mp4_boxes(data, child_payload, child_end):
                        if traf_type == b'tfhd' and traf_payload + 8 <= traf_end:
                            track_id = struct.unpack_from('>I', data, traf_payload + 4)[0]
                        elif traf_type == b'tfdt':
                            tfdt = (traf_payload, traf_end)
                    if not track_id or not tfdt or track_id not in timescales:
                        continue
                    tfdt_payload, tfdt_end = tfdt
                    version = data[tfdt_payload]
                    value_offset = tfdt_payload + 4
                    increment = round(seconds * timescales[track_id])
                    if version == 1 and value_offset + 8 <= tfdt_end:
                        current = struct.unpack_from('>Q', data, value_offset)[0]
                        struct.pack_into('>Q', data, value_offset, current + increment)
                        adjusted += 1
                    elif version == 0 and value_offset + 4 <= tfdt_end:
                        current = struct.unpack_from('>I', data, value_offset)[0]
                        next_value = current + increment
                        if next_value > 4294967295:
                            raise RuntimeError('fragment timestamp exceeds version 0 tfdt')
                        struct.pack_into('>I', data, value_offset, next_value)
                        adjusted += 1
            data.flush()
        if not adjusted:
            raise RuntimeError('fragment has no adjustable tfdt boxes')

    def _event_clip_cache_suffix(self, source_codec: str, backend: str) -> str:
        codec = source_codec or 'unknown'
        return f'a3-{backend}-{codec}'

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

    def _recording_day_rows(self, camera_id: str, start_epoch: float, end_epoch: float, source: str, *, fresh: bool=False, active_manager: AppManager | None=None) -> list[dict]:
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
                    selected_manager.recorder.lease_recordings_for_playback(cached[1])
                    return cached[1]
        rows = [row for row in selected_manager.recorder.recording_rows_between(camera_id, start_epoch, end_epoch, selected_source, discover_missing=False) if int(row.get('size_bytes') or 0) > 1024]
        if fresh:
            rows = selected_manager.recorder.discard_missing_recording_rows(rows)
        selected_manager.recorder.lease_recordings_for_playback(rows)
        selected_manager.recorder.queue_stream_fingerprints(rows)
        with self.recording_day_cache_lock:
            self.recording_day_cache[cache_key] = (now, rows)
            expired = [key for key, value in self.recording_day_cache.items() if now - value[0] >= self.recording_day_cache_seconds]
            for key in expired:
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

    def _run_recording_remux(self, command: list[str], origin: str) -> subprocess.CompletedProcess:
        if origin != 'prewarm':
            return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        with self.recording_prewarm_process_lock:
            self.recording_prewarm_process = process
        terminate_at: float | None = None
        timeout_at = time.monotonic() + 30.0
        try:
            while True:
                if not self.recording_prewarm_stop.is_set() and time.monotonic() >= timeout_at:
                    self._signal_recording_prewarm_process(process, signal.SIGTERM)
                    try:
                        _stdout, stderr = process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._signal_recording_prewarm_process(process, signal.SIGKILL)
                        _stdout, stderr = process.communicate()
                    raise subprocess.TimeoutExpired(command, 30, stderr=stderr)
                if self.recording_prewarm_stop.is_set() and process.poll() is None:
                    if terminate_at is None:
                        self._signal_recording_prewarm_process(process, signal.SIGTERM)
                        terminate_at = time.monotonic() + 3.0
                    elif time.monotonic() >= terminate_at:
                        self._signal_recording_prewarm_process(process, signal.SIGKILL)
                try:
                    _stdout, stderr = process.communicate(timeout=0.25)
                    if self.recording_prewarm_stop.is_set():
                        raise RecordingPrewarmCancelled
                    return subprocess.CompletedProcess(command, process.returncode, None, stderr)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with self.recording_prewarm_process_lock:
                if self.recording_prewarm_process is process:
                    self.recording_prewarm_process = None

    def _recording_fmp4_files(self, path: Path, duration: float, media_offset: float, origin: str='playback') -> tuple[Path, Path]:
        stat = path.stat()
        fingerprint = f'v3:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}:{media_offset:.3f}'
        cache_key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:24]
        cache_dir = self.manager.storage_dir / 'playback-cache' / 'fmp4' / cache_key
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
            command = [self.config.ffmpeg_path, '-hide_banner', '-loglevel', 'warning', '-i', str(path), '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-output_ts_offset', f'{media_offset:.3f}']
            if codec in {'hevc', 'h265'}:
                command.extend(['-tag:v', 'hvc1'])
            command.extend(['-f', 'hls', '-hls_time', '300', '-hls_list_size', '0', '-hls_segment_type', 'fmp4', '-hls_fmp4_init_filename', 'init.mp4', '-hls_segment_filename', str(temp_dir / 'media_%d.m4s'), str(temp_dir / 'index.m3u8')])
            try:
                result = self._run_recording_remux(command, origin)
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
                if time.time() - stat.st_mtime >= float(self.config.recording_segment_seconds) * 2:
                    self.manager.recorder.schedule_revalidation(path, error or 'recording fragment failed')
                with self.recording_day_cache_lock:
                    self.recording_day_cache.clear()
                raise HTTPException(status_code=500, detail=f'recording fragment failed: {error[-300:]}')
            try:
                self._offset_fmp4_timestamps(generated_init, generated_media, media_offset)
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
        active_manager: AppManager | None = None,
    ) -> Path:
        """Return a small cached JPEG near an epoch without mutating playback."""
        selected_manager = active_manager or self.manager
        selected_config = getattr(selected_manager, 'config', self.config)
        source_path = self._recording_storage_path(row.get('path'), active_manager=selected_manager)
        start_epoch = float(row.get('start_epoch') or 0)
        end_epoch = float(row.get('end_epoch') or start_epoch)
        if not start_epoch <= epoch < end_epoch:
            raise HTTPException(status_code=404, detail='no recording exists at this time')
        duration = max(0.05, end_epoch - start_epoch)
        raw_offset = max(0.0, epoch - start_epoch)
        requested_width = max(320, min(1920, int(width)))
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
                cache_dir.mkdir(parents=True, exist_ok=True)
                jpeg_quality = 3 if requested_width > 480 else 5
                command = [selected_config.ffmpeg_path, '-hide_banner', '-loglevel', 'info' if exact else 'error', '-ss', f'{preview_offset:.3f}', '-i', str(source_path), '-map', '0:v:0', '-frames:v', '1', '-threads', '1', '-vf', f"showinfo@preview,scale='min({requested_width},iw)':-2" if exact else f"scale='min({requested_width},iw)':-2", '-q:v', str(jpeg_quality), '-y', str(temporary)]
                try:
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
        return self._recording_fmp4_files(path, segment_duration, media_offset)

    def _recording_segment_path(self, camera_id: str, epoch: float, source: str='main', *, active_manager: AppManager | None=None) -> Path:
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
        return self._recording_storage_path(
            row.get('path'),
            active_manager=selected_manager,
        )

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
        return clip_dir / f'{event_id}-{safe_before}-{safe_after}-a3-{accel_mode}.mp4'

    def _ensure_event_clip(self, event: dict, *, before: float, after: float, source: str='main', active_manager: AppManager | None=None) -> Path:
        selected_manager = active_manager or self.manager
        clip_source = recording_source(source)
        clip_path = self._event_clip_path(event, before=before, after=after, source=clip_source, active_manager=selected_manager)
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
        for candidate in selected_manager.recorder.recording_rows_between(camera_id, window_start, window_end, recording_source(source), discover_missing=False):
            if candidate.get('start_epoch') is None or candidate.get('end_epoch') is None:
                continue
            try:
                candidate = {**candidate, 'path': str(self._recording_storage_path(candidate.get('path'), active_manager=selected_manager))}
            except HTTPException:
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

    def _recording_start_epoch(self, path: Path) -> float | None:
        return self.manager.recorder.recording_start_epoch(path)
