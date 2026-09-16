"""Camera execution with a single native detector/tracker and observation owner."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
import threading
import time

from .camera_capture import CameraCaptureService, FRAME_STALE_SECONDS
from .camera_lifecycle import CameraLifecyclePhase, CameraRuntimeState, CameraStopTicket
from .camera_media import CameraMediaService
from .native_activity import NativeActivity
from survng.native_spatial import spatial_plan
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)


class NativeCaptureBinding:
    """Per-camera admission preference on the shared native process."""
    def __init__(self, backend, enabled, compliance=lambda: "auto", spatial=lambda: None):
        self.backend, self.enabled = backend, enabled
        self.compliance = compliance
        self.spatial = spatial
        self.startup_timeout_ms = backend.startup_timeout_ms

    def create_handle(self):
        handle = self.backend.create_handle()
        handle.detection_enabled = bool(self.enabled())
        handle.h264_decoder_compliance = self.compliance()
        handle.spatial_plan = self.spatial()
        return handle

    def open(self, *args, **kwargs):
        return self.backend.open(*args, **kwargs)


class NativeCameraWorker:
    def __init__(self, camera, storage_dir, *, config, capture_backend, events,
                 publish, image_writer, media_storage=None, evidence_service=None):
        self.evidence_service = evidence_service
        self._last_evidence_offer = 0.0
        self.camera = camera
        self.config = config
        self.runtime_state = CameraRuntimeState()
        self._stop = self.runtime_state.stop_event
        self._lock = threading.RLock()
        self._frames_lock = threading.Lock()
        self._frames = deque(maxlen=32)
        self._thread = None
        self._enabled_at = 0.0
        self._last_native_healthy_at = time.monotonic()
        self._native_frame_session = ""
        self._observation_clock = None
        self._last_observation_epoch = 0.0
        self._last_error = ""
        self.capture = CameraCaptureService(
            camera_id=camera.id, source_url=camera.source_url, backend=NativeCaptureBinding(
                capture_backend, lambda: self.config.enabled and self.runtime_state.detection_enabled,
                lambda: self.camera.h264_decoder_compliance, lambda: spatial_plan(self.camera, self.config)),
            frame_observer=self._remember, frame_width=lambda: 0,
            initial_open_timeout_ms=capture_backend.startup_timeout_ms,
        )
        self.media = CameraMediaService(
            camera=camera, storage_dir=storage_dir, image_writer=image_writer,
            motion_detector=None, frame_provider=self._image,
            rejected_sample_rate=lambda: 0, stop_requested=self._stop.is_set,
            media_storage=media_storage, jpeg_provider=self.capture.latest_jpeg,
        )
        self.activity = NativeActivity(camera, config, events, publish, self._evidence, native_zones=True)
        self.activity.offer_evidence = self._offer_evidence
        if evidence_service is not None:
            self.activity.admission = evidence_service.admission
            self.activity.nominate = self._nominate
            self.activity.verified_snapshot = lambda cover: self.media.write_snapshot(cover[0], datetime.fromtimestamp(cover[2], timezone.utc))

    def _nominate(self, token, observation, epoch, obj):
        with self._frames_lock:
            frame = next((frame for frame in reversed(self._frames)
                          if observation.matches_frame(frame.source_pts, frame.source_session)), None)
        self.activity.admission.offer(token, self.camera.id, epoch, frame.image if frame is not None else None,
                                      obj, (observation.width, observation.height))

    def _remember(self, frame):
        if frame.source == "live":
            with self._frames_lock:
                self._frames.append(frame)
                # Retain native pixels without allowing unusually large camera
                # streams to multiply the evidence ring's memory footprint.
                total = sum(item.image.nbytes for item in self._frames)
                while len(self._frames) > 1 and total > 64 * 1024 * 1024:
                    total -= self._frames.popleft().image.nbytes

    def _offer_evidence(self, observation, epoch, event_id, objects):
        if self.evidence_service is None or epoch - self._last_evidence_offer < 1:
            return
        self._last_evidence_offer = epoch
        with self._frames_lock:
            frame = next((frame for frame in reversed(self._frames)
                          if observation.matches_frame(frame.source_pts, frame.source_session)), None)
        if frame is not None:
            self.evidence_service.offer(event_id, epoch, frame.image, objects, (observation.width, observation.height))

    def _evidence(self, observation, epoch):
        with self._frames_lock:
            frame = next((frame for frame in reversed(self._frames)
                          if observation.matches_frame(frame.source_pts, frame.source_session)), None)
        if frame is None:
            self.activity.counts["snapshot_frame_missing"] += 1
            return ""
        if (frame.width, frame.height) != (observation.width, observation.height):
            self.activity.counts["snapshot_geometry_mismatch"] += 1
            return ""
        return self.media.write_snapshot(frame.image, datetime.fromtimestamp(epoch, timezone.utc))

    def _image(self, source="live"):
        frame = self.capture.request_frame(self.camera.normalized_source(source))
        return frame.image if frame else None

    def start(self):
        with self._lock:
            if self.runtime_state.phase == CameraLifecyclePhase.CLOSED:
                raise RuntimeError("camera is closed")
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("native observation worker is still stopping")
                return
            self._stop.clear()
            self.runtime_state.enabled = True
            self.runtime_state.generation += 1
            self.runtime_state.phase = CameraLifecyclePhase.STARTING
            self._enabled_at = time.monotonic()
            self._last_native_healthy_at = self._enabled_at
            try:
                if not self.capture.start():
                    raise RuntimeError("native capture did not start")
                self._thread = threading.Thread(target=self._run, name=f"camera-{self.camera.id}-native", daemon=False)
                self._thread.start()
                self.runtime_state.phase = CameraLifecyclePhase.RUNNING
            except BaseException:
                self._stop.set()
                self.capture.request_stop()
                self.runtime_state.phase = CameraLifecyclePhase.FAILED
                raise

    def _run(self):
        last_log = 0.0
        while not self._stop.wait(0.05):
            try:
                observations = self.capture.native_observations()
                frame = self.capture.latest("live")
                now, epoch = time.monotonic(), time.time()
                with self._lock:
                    if self._stop.is_set():
                        break
                    if self.runtime_state.detection_enabled and self.config.enabled:
                        if (frame and frame.source_session
                                and frame.source_session != self._native_frame_session
                                and now - frame.captured_at_monotonic < 2):
                            # Reconnect/admission waiting is not inference time.
                            # The first resumed frame can precede its metadata;
                            # allow this new session the normal watchdog budget.
                            self._native_frame_session = frame.source_session
                            self._last_native_healthy_at = now
                        for observation in observations:
                            if observation.received_monotonic < self._enabled_at:
                                continue
                            if frame and frame.source_session == observation.session:
                                lag = frame.source_pts - observation.source_pts
                                if abs(lag) > self.config.native.maximum_observation_age_seconds:
                                    continue
                            observed_epoch = self._observation_epoch(observation, frame, now, epoch)
                            self.activity.consume(observation, now=now, epoch=observed_epoch)
                        self.activity.tick(now=now)
                        if self.activity.health == "healthy":
                            self._last_native_healthy_at = now
                        elif (frame and now - frame.captured_at_monotonic < 2
                              and now - self._last_native_healthy_at >= self.config.native.metadata_restart_seconds):
                            self._restart_native_stream(now)
                    self._last_error = ""
            except Exception as error:
                self._last_error = redact_secret_text(error)
                self.activity.health = "consumer_error"
                if time.monotonic() - last_log >= 10:
                    LOGGER.error("native observations failed for %s: %s", self.camera.id, self._last_error)
                    last_log = time.monotonic()
        with self._lock:
            try:
                self.activity.finish("stopped", now=time.monotonic())
            except Exception as error:
                self._last_error = redact_secret_text(error)
                LOGGER.error("native activity shutdown failed for %s: %s", self.camera.id, self._last_error)
                self.runtime_state.phase = CameraLifecyclePhase.FAILED

    def _observation_epoch(self, observation, frame, now, epoch):
        if self._observation_clock is None or self._observation_clock[0] != observation.session:
            estimate = epoch - max(0, now - observation.received_monotonic)
            if frame and frame.source_session == observation.session:
                estimate = frame.captured_at_epoch - (frame.source_pts - observation.source_pts)
            anchor = min(epoch, estimate)
            self._observation_clock = (observation.session, observation.source_pts, anchor)
            self._last_observation_epoch = anchor
        _, pts, anchor = self._observation_clock
        # Arrival jitter must not move an established stream's wall-time origin.
        mapped = anchor + observation.source_pts - pts
        result = max(self._last_observation_epoch, min(epoch, mapped))
        self._last_observation_epoch = result
        return result

    def _restart_native_stream(self, now):
        """Replace a stalled graph; never switch to another inference engine."""
        self.activity.finish("metadata_lost", now=now)
        self.capture.request_stop()
        if self.capture.wait_stopped(8):
            raise RuntimeError("native capture did not stop for metadata recovery")
        with self._frames_lock:
            self._frames.clear()
        self._last_native_healthy_at = time.monotonic()
        self.activity.health = "waiting_for_metadata"
        self.activity.counts["metadata_restarts"] += 1
        if not self._stop.is_set() and not self.capture.start():
            raise RuntimeError("native capture did not restart")

    def request_stop(self):
        with self._lock:
            self._stop.set()
            self.runtime_state.enabled = False
            self.runtime_state.phase = CameraLifecyclePhase.STOPPING
            ticket = CameraStopTicket(self.camera.id, self.runtime_state.generation, time.monotonic())
        self.capture.request_stop()
        return ticket

    def wait_stopped(self, deadline, ticket=None):
        if ticket is not None and ticket.generation != self.runtime_state.generation:
            return False
        if self._thread is not None:
            self._thread.join(max(0, deadline - time.monotonic()))
        remaining = self.capture.wait_stopped(max(0, deadline - time.monotonic()))
        done = not remaining and not (self._thread and self._thread.is_alive())
        if done:
            with self._lock:
                if self.runtime_state.phase != CameraLifecyclePhase.FAILED:
                    self.runtime_state.phase = CameraLifecyclePhase.STOPPED
            with self._frames_lock:
                self._frames.clear()
        return done

    def stop(self):
        ticket = self.request_stop()
        if not self.wait_stopped(time.monotonic() + 15, ticket):
            raise RuntimeError(f"native camera {self.camera.id} did not stop")

    def close(self):
        self.stop()
        self.capture.close()
        self.runtime_state.phase = CameraLifecyclePhase.CLOSED

    def active_workers(self):
        return [self._thread.name] if self._thread and self._thread.is_alive() else []

    def live_capture_ready(self):
        return self.capture.frame_ready("live")

    def set_detection_enabled(self, enabled):
        with self._lock:
            if self.runtime_state.detection_enabled == enabled:
                return
            running = self.runtime_state.enabled
        if running:
            self.stop()
        with self._lock:
            self.runtime_state.detection_enabled = enabled
            self._enabled_at = time.monotonic()
        if running:
            self.start()

    def update_zones(self, zones):
        with self._lock:
            previous = [zone.model_copy(deep=True) for zone in self.camera.zones]
            if previous == zones:
                return
            running = self.runtime_state.enabled
        if running:
            self.stop()
        try:
            with self._lock:
                self.activity.finish("zones_changed", now=time.monotonic())
                self.camera.zones = [zone.model_copy(deep=True) for zone in zones]
                self.activity.zone_revision = spatial_plan(self.camera)["revision"]
                self._enabled_at = time.monotonic()
            if running:
                self.start()
        except Exception:
            try:
                self.stop()
                with self._lock:
                    self.camera.zones = previous
                    self.activity.zone_revision = spatial_plan(self.camera)["revision"]
                if running:
                    self.start()
            except Exception:
                LOGGER.exception("failed to restore native camera zones for %s", self.camera.id)
            raise

    def reconfigure_policy(self, config):
        with self._lock:
            self.activity.finish("policy_changed", now=time.monotonic())
            self.config = config
            self.activity.config = config
            self._enabled_at = time.monotonic()

    def snapshot(self, source="live"):
        return self.media.snapshot(source)

    def mjpeg_frames(self, fps=4.0, source="live"):
        yield from self.media.mjpeg_frames(fps, source)

    # Camera fleet shutdown has a separate ONVIF release phase. Native cameras
    # own no ONVIF subscription; these acknowledge that phase without starting one.
    def request_onvif_stop(self):
        return None

    def wait_onvif_stopped(self, deadline, ticket=None):
        return True

    def stop_onvif_events(self):
        return None

    def status(self):
        capture = self.capture.status()
        now = time.monotonic()
        clock = capture.get("live_frame_monotonic")
        age = max(0, now - clock) if clock is not None else None
        connected = self.runtime_state.enabled and age is not None and age <= FRAME_STALE_SECONDS
        with self._lock:
            activity = self.activity.status()
            enabled = self.config.enabled and self.runtime_state.detection_enabled
            running = bool(enabled and self._thread and self._thread.is_alive() and not self._stop.is_set())
            activity["enabled"] = enabled
            if not enabled:
                activity["health"] = "disabled"
            elif self._stop.is_set():
                activity["health"] = "stopped"
            tracking = {**activity, "active": running, "running": running,
                        "presence_active": activity["active"]}
            return {"id": self.camera.id, "name": self.camera.name,
                    "running": self.runtime_state.enabled, "connected": connected,
                    "capture_running": capture["live_running"],
                    "capture_connectivity": "connected" if connected else "disconnected",
                    "frame_fresh": connected, "last_frame_age_seconds": age,
                    "last_frame_at": capture["live_frame_at"],
                    "last_error": self._last_error or capture["last_error"],
                    "detection_enabled": self.runtime_state.detection_enabled and self.config.enabled,
                    "onvif_enabled": False, "onvif_connected": False,
                    "last_motion_at": self.activity.last_motion_at,
                    "object_tracking": tracking, "native_activity": activity,
                    "motion_qualification": {"mode": "native", "enabled": False},
                    "lifecycle": {"phase": self.runtime_state.phase.value, "generation": self.runtime_state.generation},
                    **{key: capture.get(key) for key in ("capture_stats", "live_pipeline", "live_detection_matching", "live_detections", "stream_dimensions", "main_running", "main_last_error")}}
