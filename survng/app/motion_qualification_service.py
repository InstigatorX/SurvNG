"""Camera-scoped motion qualification, evidence fusion, and replay policy."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np

from .config import CameraConfig, MotionQualificationConfig
from .motion import MotionQualificationResult
from .motion_coordinator import VisualBackupPolicy
from .motion_pipeline import (
    MotionContext,
    MotionDebugSnapshotStore,
    MotionPipeline,
    MotionScoring,
    resolved_trigger_mode,
)

LOGGER = logging.getLogger(__name__)
FUSION_STALE_TOLERANCE_SECONDS = 5.0


class MotionQualificationState(Protocol):
    def increment_stat(self, name: str, amount: int = 1) -> None: ...


class MotionSampleSource(Protocol):
    def samples_since(self, captured_at: float) -> list[tuple[float, np.ndarray]]: ...

    def qualification_results_since(
        self,
        captured_at: float,
    ) -> list[tuple[float, MotionQualificationResult]]: ...


class MotionQualificationService:
    """Own qualification configuration, pipeline execution, and fusion state."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        config: MotionQualificationConfig,
        qualification_pipeline: MotionPipeline,
        observation_pipeline: MotionPipeline,
        fusion_pipeline: MotionPipeline,
        pipeline_origins: dict[str, str],
        debug_store: MotionDebugSnapshotStore,
        stop_event: threading.Event,
        state: MotionQualificationState,
    ) -> None:
        self.camera = camera
        self.config = config
        self.qualification_pipeline = qualification_pipeline
        self.observation_pipeline = observation_pipeline
        self.fusion_pipeline = fusion_pipeline
        self.pipeline_origins = dict(pipeline_origins)
        self.debug_store = debug_store
        self.stop_event = stop_event
        self.state = state
        self.analysis_lock = threading.Lock()
        self._observation_lock = threading.Lock()
        self._fusion_lock = threading.Lock()
        self._fusion_last_at = 0.0
        self._pipeline_configuration = {
            **self.config.model_dump(mode="python"),
            "camera_id": self.camera.id,
            "motion_zones": [
                zone.model_dump(mode="python") for zone in self.camera.zones
            ],
        }

    def settings(self) -> tuple[str, str, int]:
        override = self.camera.motion_qualification
        mode = self.config.mode if override.mode == "inherit" else override.mode
        sensitivity = (
            self.config.sensitivity
            if override.sensitivity == "inherit"
            else override.sensitivity
        )
        return mode, sensitivity, int(override.frame_width or self.config.frame_width)

    def stationary_object_tolerance(self) -> str:
        override = self.camera.motion_qualification.stationary_object_tolerance
        return self.config.stationary_object_tolerance if override == "inherit" else override

    def visual_backup_settings(self) -> dict[str, float | int]:
        override = self.camera.motion_qualification
        return {
            "grace_seconds": float(
                self.config.visual_backup_grace_seconds
                if override.visual_backup_grace_seconds is None
                else override.visual_backup_grace_seconds
            ),
            "minimum_score": float(
                self.config.visual_backup_min_score
                if override.visual_backup_min_score is None
                else override.visual_backup_min_score
            ),
            "minimum_consecutive": int(
                self.config.visual_backup_min_consecutive
                if override.visual_backup_min_consecutive is None
                else override.visual_backup_min_consecutive
            ),
            "cooldown_seconds": float(
                self.config.visual_backup_cooldown_seconds
                if override.visual_backup_cooldown_seconds is None
                else override.visual_backup_cooldown_seconds
            ),
            "maximum_triggers_5m": int(
                self.config.visual_backup_max_triggers_5m
                if override.visual_backup_max_triggers_5m is None
                else override.visual_backup_max_triggers_5m
            ),
        }

    def visual_backup_policy(self) -> VisualBackupPolicy:
        settings = self.visual_backup_settings()
        return VisualBackupPolicy(
            warmup_seconds=self.config.visual_backup_warmup_seconds,
            grace_seconds=float(settings["grace_seconds"]),
            minimum_score=float(settings["minimum_score"]),
            score_margin=self.config.visual_backup_score_margin,
            minimum_consecutive=int(settings["minimum_consecutive"]),
            cooldown_seconds=float(settings["cooldown_seconds"]),
            maximum_triggers_5m=int(settings["maximum_triggers_5m"]),
            sample_fps=self.config.sample_fps,
            background_fps=self.config.camera_mode_background_fps,
        )

    def trigger_mode(self) -> str:
        return resolved_trigger_mode(self.settings()[0])

    def fusion_options(self) -> dict[str, Any]:
        return next(
            (
                dict(stage.get("options") or {})
                for stage in self.fusion_pipeline.stage_configuration
                if stage.get("implementation") == "buffered_evidence_fusion"
            ),
            {},
        )

    def adaptive_analysis_required(self) -> bool:
        if self.trigger_mode() in {"adaptive", "camera_rescue"}:
            return True
        options = self.fusion_options()
        return bool(
            str(options.get("policy", "audit")).strip().lower() != "bypass"
            and options.get("include_primary", True)
        )

    def continuous_primary_required(self) -> bool:
        return bool(
            self.adaptive_analysis_required()
            and (
                self.qualification_pipeline.continuous_analysis
                or self.trigger_mode() in {"adaptive", "camera_rescue"}
            )
        )

    def preprocessor_implementation(self) -> str:
        """Return the configured qualification preprocessor implementation."""
        return next(
            (
                str(stage.get("implementation") or "")
                for stage in self.qualification_pipeline.stage_configuration
                if str(stage.get("stage_id") or "") == "preprocess"
            ),
            "",
        )

    def continuous_primary_due(
        self,
        captured_at: float,
        last_processed_at: float,
    ) -> bool:
        if self.trigger_mode() == "adaptive":
            return True
        background_fps = min(
            self.config.sample_fps,
            self.config.camera_mode_background_fps,
        )
        interval = 1.0 / max(0.5, background_fps)
        return bool(
            last_processed_at <= 0.0
            or captured_at - last_processed_at >= interval * 0.85
        )

    def external_confirmation_required(self) -> bool:
        options = self.fusion_options()
        raw_sources = options.get("sources", [])
        source_values = (raw_sources,) if isinstance(raw_sources, str) else raw_sources
        sources = (
            tuple(
                normalized
                for source in source_values
                if (normalized := str(source).strip().lower())
            )
            if isinstance(source_values, (list, tuple))
            else ()
        )
        policy = str(options.get("policy", "audit")).strip().lower()
        return bool(
            sources
            and (
                policy in {"all", "weighted"}
                or not bool(options.get("include_primary", True))
            )
        )

    def frame_analysis_required(self) -> bool:
        return bool(
            self.adaptive_analysis_required()
            or self.observation_pipeline.handles_observation("frame")
            or self.debug_store.enabled()
        )

    def rescue_settings(self) -> tuple[bool, float]:
        override = self.camera.motion_qualification
        enabled = (
            self.config.borderline_rescue_enabled
            if override.borderline_rescue_enabled is None
            else override.borderline_rescue_enabled
        )
        margin = (
            self.config.borderline_margin
            if override.borderline_margin is None
            else override.borderline_margin
        )
        return bool(enabled), float(margin)

    def suppression_verification_rate(self) -> float:
        override = self.camera.motion_qualification.suppression_verification_rate
        return float(
            self.config.suppression_verification_rate
            if override is None
            else override
        )

    def illumination_filter_enabled(self) -> bool:
        override = self.camera.motion_qualification.illumination_filter_enabled
        return bool(
            self.config.illumination_filter_enabled
            if override is None
            else override
        )

    def observe_event(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        received_at: float,
    ) -> None:
        try:
            with self._observation_lock:
                if self.observation_pipeline.handles_observation("motion_event"):
                    self.observation_pipeline.process(MotionContext(
                        camera_id=self.camera.id,
                        captured_at=received_at,
                        original_frame=None,
                        configuration={
                            "observation_kind": "motion_event",
                            "event_source": (
                                "manual" if topic.startswith("manual") else "onvif"
                            ),
                            "event_topic": topic,
                            "event_message": message,
                            "event_at": event_at.timestamp(),
                        },
                        runtime=self.observation_pipeline.runtime,
                    ))
        except Exception as error:
            LOGGER.warning(
                "motion event evidence failed for %s: %s",
                self.camera.id,
                error,
            )

    def observe_frame(self, frame: np.ndarray, captured_at: float) -> None:
        """Apply frame observations under the observation runtime's own lock."""
        with self._observation_lock:
            if not self.observation_pipeline.handles_observation("frame"):
                return
            self.observation_pipeline.process(MotionContext(
                camera_id=self.camera.id,
                captured_at=captured_at,
                original_frame=frame,
                configuration={"observation_kind": "frame"},
                runtime=self.observation_pipeline.runtime,
            ))

    def with_source_evidence(
        self,
        result: MotionQualificationResult,
        start_epoch: float,
        end_epoch: float,
        *,
        include_telemetry: bool = True,
        require_primary_trigger: bool = False,
    ) -> MotionQualificationResult:
        with self._fusion_lock:
            stale_by = self._fusion_last_at - end_epoch
            if 0.0 < stale_by <= FUSION_STALE_TOLERANCE_SECONDS:
                self.state.increment_stat("stale_fusion_samples", 1)
                return MotionQualificationResult(
                    accepted=False,
                    score=0.0,
                    threshold=1.0,
                    reason="stale_fusion_evidence",
                    frame_count=result.frame_count,
                    features={
                        **result.features,
                        "stale_fusion_seconds": round(stale_by, 3),
                        "stale_fusion_original_score": result.score,
                    },
                    telemetry=dict(result.telemetry),
                )
            context = MotionContext(
                camera_id=self.camera.id,
                captured_at=end_epoch,
                original_frame=None,
                configuration={
                    "evidence_started_at": start_epoch,
                    "evidence_ended_at": end_epoch,
                    "require_primary_trigger": require_primary_trigger,
                },
                runtime=self.fusion_pipeline.runtime,
                scoring=MotionScoring(
                    accepted=result.accepted,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    frame_count=result.frame_count,
                    features=dict(result.features),
                ),
            )
            try:
                processed = self.fusion_pipeline.process(context)
            except Exception as error:
                return self.validation_fail_open_result(
                    "motion fusion pipeline",
                    error,
                    result,
                    allow_detection=not (
                        require_primary_trigger and not result.accepted
                    ),
                )
            self._fusion_last_at = end_epoch
        scoring = processed.scoring
        decision = processed.decision
        features = dict(scoring.features)
        features.update({
            "event_state_phase": processed.event_state.phase.value,
            "event_state_key": processed.event_state.event_key,
            "event_state_started_at": processed.event_state.started_at,
            "event_state_transition": processed.event_state.transition_reason,
            "event_state_consecutive_accepts": processed.event_state.consecutive_accepts,
            "event_state_consecutive_rejects": processed.event_state.consecutive_rejects,
        })
        if processed.event_state.cooldown_until is not None:
            features["event_state_cooldown_remaining"] = round(
                max(0.0, processed.event_state.cooldown_until - end_epoch), 3
            )
        telemetry = dict(result.telemetry)
        if include_telemetry:
            graphs = dict(telemetry.get("graphs") or {})
            graphs.setdefault(
                "qualification", self.qualification_pipeline.audit_snapshot()
            )
            graphs["observation"] = self.observation_pipeline.audit_snapshot()
            graphs["fusion"] = self.fusion_pipeline.audit_snapshot(processed.timings)
            telemetry.update({
                "schema_version": 1,
                "origins": dict(self.pipeline_origins),
                "graphs": graphs,
            })
        return MotionQualificationResult(
            accepted=(
                decision.run_object_detection if decision is not None else scoring.accepted
            ),
            score=decision.score if decision is not None else scoring.score,
            threshold=scoring.threshold,
            reason=decision.reason if decision is not None else scoring.reason,
            frame_count=scoring.frame_count,
            features=features,
            telemetry=telemetry,
        )

    def reset_event_state_runtime(self) -> None:
        stage_ids = frozenset(
            str(stage.get("stage_id") or "")
            for stage in self.fusion_pipeline.stage_configuration
            if stage.get("implementation") == "score_event_state"
        )
        with self._fusion_lock:
            self.fusion_pipeline.runtime.reset_stages(stage_ids)

    def reset_runtime(
        self,
        *,
        clear_observation_evidence: Callable[[], None] | None = None,
    ) -> None:
        # Observation and qualification stages have independent execution
        # lanes. Reset each runtime under the lock used by its own lane so a
        # native/GPU stage is never closed while it is processing.
        with self._observation_lock:
            if clear_observation_evidence is not None:
                clear_observation_evidence()
            self.observation_pipeline.runtime.reset()
        with self.analysis_lock:
            self.qualification_pipeline.runtime.reset()
        with self._fusion_lock:
            self.fusion_pipeline.runtime.reset()
            self._fusion_last_at = 0.0

    def validation_fail_open_result(
        self,
        component: str,
        error: Exception,
        original: MotionQualificationResult | None = None,
        *,
        allow_detection: bool = True,
    ) -> MotionQualificationResult:
        self.state.increment_stat("validation_failures", 1)
        self.state.increment_stat("validation_fail_opens", int(allow_detection))
        LOGGER.error(
            "%s unavailable for %s; %s (%s)",
            component,
            self.camera.id,
            (
                "allowing object detection"
                if allow_detection
                else "preserving rejected primary trigger"
            ),
            type(error).__name__,
        )
        features = dict(original.features) if original is not None else {}
        features.update({
            "validation_unavailable": True,
            "validation_fail_open": allow_detection,
            "validation_failure_component": component,
            "validation_failure_type": type(error).__name__,
        })
        return MotionQualificationResult(
            allow_detection,
            1.0 if allow_detection else (original.score if original else 0.0),
            0.0 if allow_detection else (original.threshold if original else 1.0),
            (
                "validation_unavailable_fail_open"
                if allow_detection
                else "primary_trigger_rejected"
            ),
            original.frame_count if original else 0,
            features,
            telemetry=dict(original.telemetry) if original else {},
        )

    def with_pipeline_telemetry(
        self, result: MotionQualificationResult
    ) -> MotionQualificationResult:
        if result.telemetry:
            return result
        return MotionQualificationResult(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features=dict(result.features),
            telemetry={
                "schema_version": 1,
                "origins": dict(self.pipeline_origins),
                "graphs": {
                    "qualification": self.qualification_pipeline.audit_snapshot(),
                    "observation": self.observation_pipeline.audit_snapshot(),
                    "fusion": self.fusion_pipeline.audit_snapshot(),
                },
            },
        )

    def qualify_burst(
        self,
        event_at: datetime,
        received_at: float,
        sensitivity: str,
        sample_source: MotionSampleSource,
    ) -> tuple[MotionQualificationResult, dict[str, Any]]:
        event_epoch = event_at.timestamp()
        anchor = (
            min(event_epoch, received_at)
            if abs(event_epoch - received_at) <= 10.0
            else received_at
        )
        if not self.adaptive_analysis_required():
            if self.external_confirmation_required():
                self.stop_event.wait(self.config.post_trigger_seconds)
            result = MotionQualificationResult(
                False, 0.0, 1.0, "adaptive_validation_disabled", 0, {}
            )
            return self.with_source_evidence(
                result,
                anchor - self.config.window_seconds,
                time.time(),
            ), self._diagnostics(received_at, event_epoch, 0)

        deadline = time.monotonic() + self.config.post_trigger_seconds
        best_result: MotionQualificationResult | None = None
        evaluated_windows: set[tuple[float, ...]] = set()
        samples: list[tuple[float, np.ndarray]] = []
        result_source = getattr(sample_source, "qualification_results_since", None)
        while not self.stop_event.is_set():
            computed: list[tuple[float, MotionQualificationResult]] = []
            if callable(result_source):
                computed = result_source(received_at)
                for result_at, result in computed:
                    key = (round(float(result_at), 3),)
                    if key in evaluated_windows:
                        continue
                    evaluated_windows.add(key)
                    if best_result is None or result.score > best_result.score:
                        best_result = result
                    if result.accepted and not self.external_confirmation_required():
                        return self.with_source_evidence(
                            result,
                            anchor - self.config.window_seconds,
                            time.time(),
                        ), self._diagnostics(
                            received_at, event_epoch, len(evaluated_windows)
                        )
            samples = sample_source.samples_since(
                anchor - self.config.window_seconds
            )
            # Production supplies results from its continuously advancing EMA
            # runtime. The clean-runtime replay is retained for deterministic
            # tools and alternate sample providers only.
            replay_indices = (
                range(3, len(samples)) if not computed else ()
            )
            for end_index in replay_indices:
                window_end = samples[end_index][0]
                if window_end < received_at:
                    continue
                window_start = window_end - self.config.window_seconds
                window = [
                    item for item in samples[: end_index + 1] if item[0] >= window_start
                ]
                if (
                    len(window) < 4
                    or window[-1][0] - window[0][0]
                    < self.config.window_seconds * 0.45
                ):
                    continue
                key = tuple(round(item[0], 3) for item in window)
                if key in evaluated_windows:
                    continue
                evaluated_windows.add(key)
                try:
                    result = self.run_pipeline(
                        [item[1] for item in window],
                        sensitivity,
                        window_end,
                        [item[0] for item in window],
                        clone_runtime=False,
                    )
                except Exception as error:
                    return self.validation_fail_open_result(
                        "adaptive validation pipeline", error
                    ), self._diagnostics(
                        received_at, event_epoch, len(evaluated_windows)
                    )
                if best_result is None or result.score > best_result.score:
                    best_result = result
                if result.accepted and not self.external_confirmation_required():
                    return self.with_source_evidence(
                        result,
                        anchor - self.config.window_seconds,
                        time.time(),
                    ), self._diagnostics(
                        received_at, event_epoch, len(evaluated_windows)
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.stop_event.wait(min(0.2, remaining)):
                break

        diagnostics = self._diagnostics(
            received_at, event_epoch, len(evaluated_windows)
        )
        if best_result is None:
            result = MotionQualificationResult(
                True, 1.0, 0.0, "insufficient_frames", len(samples), {}
            )
        elif best_result.score == 0.0 and not best_result.features.get("global_change"):
            result = MotionQualificationResult(
                True,
                0.0,
                best_result.threshold,
                "no_temporal_signal",
                best_result.frame_count,
                best_result.features,
            )
        else:
            result = best_result
        return self.with_source_evidence(
            result,
            anchor - self.config.window_seconds,
            time.time(),
        ), diagnostics

    @staticmethod
    def _diagnostics(
        received_at: float, event_epoch: float, windows: int
    ) -> dict[str, Any]:
        return {
            "windows_evaluated": windows,
            "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
        }

    def run_pipeline(
        self,
        frames: list[np.ndarray],
        sensitivity: str,
        captured_at: float,
        frame_timestamps: list[float] | None = None,
        *,
        isolated: bool = True,
        capture_debug: bool = True,
        include_telemetry: bool = True,
        processed_frames: list[np.ndarray] | None = None,
        processed_frame_implementation: str = "",
        clone_runtime: bool = True,
    ) -> MotionQualificationResult:
        if (
            processed_frames is not None
            and processed_frame_implementation != self.preprocessor_implementation()
        ):
            processed_frames = None
        if processed_frames is not None and len(processed_frames) != len(frames):
            raise ValueError(
                "processed_frames must contain one derivative for each source frame"
            )
        mode, _resolved_sensitivity, frame_width = self.settings()
        if isolated:
            with self.analysis_lock:
                pipeline = self.qualification_pipeline.isolated_copy(
                    clone_runtime=clone_runtime
                )
        else:
            pipeline = self.qualification_pipeline
        context = MotionContext(
            camera_id=self.camera.id,
            captured_at=captured_at,
            original_frame=frames[-1] if frames else None,
            frame_history=tuple(frames),
            frame_timestamps=tuple(frame_timestamps or ()),
            processed_frame_history=tuple(processed_frames or ()),
            processed_frame=(processed_frames[-1] if processed_frames else None),
            configuration={
                **self._pipeline_configuration,
                "mode": mode,
                "sensitivity": sensitivity,
                "stationary_object_tolerance": self.stationary_object_tolerance(),
                "illumination_filter_enabled": self.illumination_filter_enabled(),
                "frame_width": frame_width,
            },
            runtime=pipeline.runtime,
        )
        try:
            processed = pipeline.process(context)
        finally:
            if isolated:
                pipeline.close()
        if capture_debug:
            self.debug_store.capture(processed)
        scoring = processed.scoring
        features = dict(scoring.features)
        features.setdefault(
            "primary_motion_source", self.qualification_pipeline.primary_motion_source
        )
        dominant = processed.dominant_track
        if dominant is not None and dominant.observations:
            features.setdefault(
                "motion_regions",
                [
                    [round(float(value), 5) for value in blob.box]
                    for blob in dominant.observations[-12:]
                ],
            )
            features.setdefault("motion_region_track_id", dominant.track_id)
        telemetry: dict[str, Any] = {}
        if include_telemetry:
            telemetry = {
                "schema_version": 1,
                "origins": dict(self.pipeline_origins),
                "graphs": {
                    "qualification": pipeline.audit_snapshot(processed.timings)
                },
            }
        return MotionQualificationResult(
            accepted=scoring.accepted,
            score=scoring.score,
            threshold=scoring.threshold,
            reason=scoring.reason,
            frame_count=scoring.frame_count,
            features=features,
            telemetry=telemetry,
        )

    def set_debug_enabled(self, enabled: bool) -> None:
        self.debug_store.set_enabled(enabled)

    def debug_status(self) -> dict[str, Any]:
        return self.debug_store.status()

    def debug_image(self, layer: str) -> bytes | None:
        return self.debug_store.image(layer)
