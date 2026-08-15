"""Transactional ownership for inference, tracking, and similarity services."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from .appearance_backfill import DeferredAppearanceBackfill
from .appearance_index import AppearanceIndex
from .config import DetectorConfig, SemanticSearchConfig
from .events import EventStore
from .faces import FaceStore
from .inference import (
    InferenceSupervisor,
    IsolatedFaceRecognizer,
    IsolatedPersonReidentifier,
)
from .object_tracking import (
    AdaptiveTrackingLimiter,
    ObjectTrackingSession,
    ObjectTrackingSessionFactory,
)
from .semantic_search import DisabledSemanticSearch, SemanticIndex, build_semantic_search
from .media_storage import MediaStorageRegistry
from .security import redact_secret_text

LOGGER = logging.getLogger("uvicorn.error")


class TrackingWorker(Protocol):
    camera: object

    def create_object_tracking_session(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession: ...

    def replace_object_tracking_session(
        self,
        replacement: ObjectTrackingSession,
    ) -> ObjectTrackingSession: ...

    def pause_object_tracking_session(self) -> None: ...
    def resume_object_tracking_session(self) -> None: ...


class InferenceLifecycle:
    """Own one replaceable inference/tracking/search runtime generation."""

    def __init__(
        self,
        *,
        config: DetectorConfig,
        semantic_config: SemanticSearchConfig,
        storage_dir: Path,
        events: EventStore,
        appearance_index: AppearanceIndex,
        semantic_index: SemanticIndex,
        event_publisher: Callable[[str, dict], None],
        tracking_burst_guard: Callable[[], bool],
        database_dir: Path,
        media_storage: MediaStorageRegistry | None = None,
    ) -> None:
        self.storage_dir = storage_dir
        self.events = events
        self.appearance_index = appearance_index
        self.semantic_index = semantic_index
        self.event_publisher = event_publisher
        self.tracking_burst_guard = tracking_burst_guard
        self.database_dir = database_dir
        self.media_storage = media_storage
        self.detector = InferenceSupervisor(config)
        faces: FaceStore | None = None
        semantic_search: DisabledSemanticSearch | None = None
        appearance_backfill: DeferredAppearanceBackfill | None = None
        try:
            self.face_recognizer = IsolatedFaceRecognizer(self.detector)
            self.person_reidentifier = IsolatedPersonReidentifier(self.detector)
            faces = FaceStore(
                storage_dir,
                config.face_max_observations,
                self.face_recognizer,
                start_recognition=False,
                database_dir=database_dir,
                media_storage=media_storage,
            )
            semantic_search = build_semantic_search(
                semantic_config,
                semantic_index,
            )
            self.tracking_limiter = self._build_limiter(config)
            self.tracking_factory = self._build_tracking_factory(
                config,
                self.tracking_limiter,
            )
            appearance_backfill = self._build_backfill(config)
        except BaseException:
            for operation in (
                getattr(appearance_backfill, "close", None),
                getattr(semantic_search, "close", None),
                getattr(faces, "close", None),
                self.detector.stop,
            ):
                if operation is None:
                    continue
                try:
                    operation()
                except Exception as error:
                    LOGGER.error(
                        "inference construction rollback failed: %s",
                        redact_secret_text(error),
                    )
            raise
        self.faces = faces
        self.semantic_search = semantic_search
        self.appearance_backfill = appearance_backfill
        self._workers: dict[str, TrackingWorker] = {}
        self._workers_bound = False
        self._lock = threading.RLock()
        self._core_started = False
        self._auxiliary_started = False
        self._closed = False
        self._retired_cleanup: list[tuple[str, Callable[[], object]]] = []

    def bind_workers(self, workers: Mapping[str, TrackingWorker]) -> None:
        """Bind the camera generation once after dependency construction."""
        with self._lock:
            if self._workers_bound:
                raise RuntimeError("inference lifecycle workers are already bound")
            if self._core_started or self._closed:
                raise RuntimeError("cannot bind workers after inference lifecycle start")
            self._workers = dict(workers)
            self._workers_bound = True

    def replace_worker(self, camera_id: str, worker: TrackingWorker) -> None:
        with self._lock:
            if not self._workers_bound or camera_id not in self._workers:
                raise RuntimeError(f"camera {camera_id} is not bound")
            self._workers[camera_id] = worker

    def start_core(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("inference lifecycle is closed")
            if self._core_started:
                return
            self.detector.start()
            try:
                self.faces.start()
            except BaseException:
                for label, operation in (
                    ("face queue", self.faces.close),
                    ("detector", self.detector.stop),
                ):
                    try:
                        operation()
                    except Exception as error:
                        LOGGER.error(
                            "%s rollback failed after inference startup error: %s",
                            label,
                            redact_secret_text(error),
                        )
                raise
            self._core_started = True

    def start_auxiliary(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("inference lifecycle is closed")
            if self._auxiliary_started:
                return
            self.appearance_backfill.start()
            try:
                self.semantic_search.start(
                    self.events,
                    self.storage_dir,
                    getattr(self, "media_storage", None),
                )
            except BaseException:
                for label, operation in (
                    ("semantic search", self.semantic_search.close),
                    ("appearance backfill", self.appearance_backfill.close),
                ):
                    try:
                        operation()
                    except Exception as error:
                        LOGGER.error(
                            "%s rollback failed after auxiliary startup error: %s",
                            label,
                            redact_secret_text(error),
                        )
                raise
            self._auxiliary_started = True

    def close(self) -> None:
        """Close all inference-owned services, attempting every component."""
        with self._lock:
            if self._closed:
                return
            failures: list[tuple[str, BaseException]] = []
            for label, operation in (
                ("face recognition", self.faces.close),
                ("semantic search", self.semantic_search.close),
                ("appearance backfill", self.appearance_backfill.close),
                *tuple(self._retired_cleanup),
                ("inference", self.detector.stop),
            ):
                try:
                    operation()
                except BaseException as error:
                    failures.append((label, error))
                    LOGGER.error(
                        "%s shutdown failed: %s",
                        label,
                        redact_secret_text(error),
                    )
            self._core_started = False
            self._auxiliary_started = False
            self._closed = True
            self._retired_cleanup = []
            if failures:
                labels = ", ".join(label for label, _error in failures)
                first = failures[0][1]
                if not isinstance(first, Exception):
                    raise first
                raise RuntimeError(
                    f"inference lifecycle shutdown failed: {labels}"
                ) from None

    def reconfigure_policy(self, config: DetectorConfig) -> None:
        with self._lock:
            self._ensure_open()
            previous = self.detector.config.model_copy(deep=True)
            refresh_faces = (
                previous.face_match_threshold != config.face_match_threshold
                or previous.face_max_references != config.face_max_references
            )
            try:
                self.detector.update_runtime_config(config)
                self.faces.reconfigure_max_observations(config.face_max_observations)
            except BaseException as error:
                try:
                    self.detector.update_runtime_config(previous)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "detector policy rollback failed: "
                        f"{redact_secret_text(rollback_error)}"
                    ) from error
                raise
            if refresh_faces:
                self.faces.request_match_refresh()

    def reconfigure_tracking(self, config: DetectorConfig) -> None:
        """Transactionally replace every camera tracking session and backfill."""
        with self._lock:
            self._ensure_open()
            tracking = config.tracking.model_copy(deep=True)
            next_limiter = self._build_limiter(config)
            next_factory = self._build_tracking_factory(config, next_limiter)
            workers = list(self._workers.values())
            replacements: list[ObjectTrackingSession] = []
            try:
                for worker in workers:
                    replacements.append(
                        worker.create_object_tracking_session(next_factory)
                    )
                next_backfill = self._build_backfill(config)
            except BaseException:
                self._stop_sessions(replacements)
                raise
            previous_sessions: list[tuple[TrackingWorker, ObjectTrackingSession]] = []
            previous_config = self.detector.config.model_copy(deep=True)
            previous_factory = self.tracking_factory
            previous_limiter = self.tracking_limiter
            previous_backfill = self.appearance_backfill
            try:
                for worker, replacement in zip(workers, replacements, strict=True):
                    previous = worker.replace_object_tracking_session(replacement)
                    previous_sessions.append((worker, previous))
                self.detector.update_runtime_config(config)
                self.person_reidentifier.config = self.detector.config.tracking
                self.tracking_factory = next_factory
                self.tracking_limiter = next_limiter
                self.appearance_backfill = next_backfill
                if self._auxiliary_started:
                    next_backfill.start()
            except BaseException:
                rejected_sessions: list[ObjectTrackingSession] = list(
                    replacements[len(previous_sessions):]
                )
                for worker, previous in reversed(previous_sessions):
                    try:
                        rejected_sessions.append(
                            worker.replace_object_tracking_session(previous)
                        )
                    except Exception as error:
                        camera = getattr(worker, "camera", None)
                        LOGGER.error(
                            "failed to roll back object tracking for %s: %s",
                            getattr(camera, "id", "unknown"),
                            redact_secret_text(error),
                        )
                self._stop_sessions(rejected_sessions)
                try:
                    self.detector.update_runtime_config(previous_config)
                except Exception as error:
                    LOGGER.error(
                        "failed to restore detector tracking configuration: %s",
                        redact_secret_text(error),
                    )
                self.person_reidentifier.config = self.detector.config.tracking
                self.tracking_factory = previous_factory
                self.tracking_limiter = previous_limiter
                self.appearance_backfill = previous_backfill
                try:
                    next_backfill.close()
                except Exception as error:
                    LOGGER.error(
                        "failed to close rejected appearance backfill: %s",
                        redact_secret_text(error),
                    )
                raise
            self._retire("previous appearance backfill", previous_backfill.close)

    def reconfigure_roles(
        self,
        config: DetectorConfig,
        roles: set[str],
        *,
        refresh_tracking: bool = False,
    ) -> None:
        """Replace selected inference roles with complete rollback on failure."""
        with self._lock:
            self._ensure_open()
            previous_config = self.detector.config.model_copy(deep=True)
            face_role = "face" in roles
            inference_applied = False
            tracking_applied = False
            face_queue_stop_attempted = False
            try:
                if refresh_tracking:
                    self._pause_tracking()
                if face_role:
                    face_queue_stop_attempted = True
                    self.faces.close()
                self.detector.reconfigure_roles(config, roles)
                inference_applied = True
                self.person_reidentifier.config = self.detector.config.tracking
                if refresh_tracking:
                    self.reconfigure_tracking(config)
                    tracking_applied = True
                if face_role:
                    self.faces.start()
            except BaseException:
                if refresh_tracking:
                    self._pause_tracking(best_effort=True)
                inference_restored = not inference_applied
                if inference_applied:
                    try:
                        self.detector.reconfigure_roles(previous_config, roles)
                        inference_restored = True
                    except Exception as error:
                        LOGGER.error(
                            "failed to roll back inference roles: %s",
                            redact_secret_text(error),
                        )
                if tracking_applied and inference_restored:
                    try:
                        self.reconfigure_tracking(previous_config)
                    except Exception as error:
                        LOGGER.error(
                            "failed to roll back object tracking sessions: %s",
                            redact_secret_text(error),
                        )
                elif refresh_tracking and inference_restored:
                    self._resume_tracking(best_effort=True)
                self.person_reidentifier.config = self.detector.config.tracking
                if face_queue_stop_attempted:
                    try:
                        self.faces.start()
                    except Exception as error:
                        LOGGER.error(
                            "failed to restore face recognition queue: %s",
                            redact_secret_text(error),
                        )
                raise

    def reconfigure_semantic_search(self, config: SemanticSearchConfig) -> None:
        with self._lock:
            self._ensure_open()
            replacement = build_semantic_search(config, self.semantic_index)
            try:
                if self._auxiliary_started:
                    replacement.start(self.events, self.storage_dir)
            except BaseException:
                try:
                    replacement.close()
                except Exception as error:
                    LOGGER.error(
                        "failed to close rejected semantic search: %s",
                        redact_secret_text(error),
                    )
                raise
            previous = self.semantic_search
            self.semantic_search = replacement
            self._retire("previous semantic search", previous.close)

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "core_started": self._core_started,
                "auxiliary_started": self._auxiliary_started,
                "closed": self._closed,
                "bound_cameras": len(self._workers),
                "workers_bound": self._workers_bound,
                "retired_cleanup_pending": len(self._retired_cleanup),
            }

    def maintain(self) -> None:
        """Retry retirement cleanup without changing the active generation."""
        with self._lock:
            pending = self._retired_cleanup
            self._retired_cleanup = []
            for label, operation in pending:
                self._retire(label, operation)

    def _build_limiter(self, config: DetectorConfig) -> AdaptiveTrackingLimiter:
        tracking = config.tracking
        return AdaptiveTrackingLimiter(
            tracking.max_active_cameras,
            tracking.burst_max_active_cameras,
            burst_enabled=tracking.adaptive_burst_enabled,
            burst_guard=self.tracking_burst_guard,
        )

    def _build_tracking_factory(
        self,
        config: DetectorConfig,
        limiter: AdaptiveTrackingLimiter,
    ) -> ObjectTrackingSessionFactory:
        return ObjectTrackingSessionFactory(
            config=config.tracking,
            detector=self.detector,
            update_event=self.events.update_object_tracking,
            publisher=self.event_publisher,
            limiter=limiter,
            appearance_encoder=self.person_reidentifier,
            appearance_indexer=self.appearance_index.replace_event,
            cover_promoter=self.events.promote_tracking_cover,
        )

    def _build_backfill(self, config: DetectorConfig) -> DeferredAppearanceBackfill:
        return DeferredAppearanceBackfill(
            self.events.db_path,
            self.storage_dir,
            config.tracking,
            self.events,
            self.appearance_index,
            self.person_reidentifier,
            media_storage=self.media_storage,
        )

    def _pause_tracking(self, *, best_effort: bool = False) -> None:
        for worker in self._workers.values():
            try:
                worker.pause_object_tracking_session()
            except Exception as error:
                if not best_effort:
                    raise
                camera = getattr(worker, "camera", None)
                LOGGER.error(
                    "failed to quiesce object tracking for %s during rollback: %s",
                    getattr(camera, "id", "unknown"),
                    redact_secret_text(error),
                )

    def _resume_tracking(self, *, best_effort: bool = False) -> None:
        for worker in self._workers.values():
            try:
                worker.resume_object_tracking_session()
            except Exception as error:
                if not best_effort:
                    raise
                camera = getattr(worker, "camera", None)
                LOGGER.error(
                    "failed to resume object tracking for %s: %s",
                    getattr(camera, "id", "unknown"),
                    redact_secret_text(error),
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("inference lifecycle is closed")

    def _retire(self, label: str, operation: Callable[[], object]) -> None:
        try:
            operation()
        except Exception as error:
            LOGGER.error(
                "%s cleanup failed; retaining for shutdown retry: %s",
                label,
                redact_secret_text(error),
            )
            self._retired_cleanup.append((label, operation))

    @staticmethod
    def _stop_sessions(sessions: list[ObjectTrackingSession]) -> None:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception as error:
                LOGGER.error(
                    "failed to close unbound tracking session: %s",
                    redact_secret_text(error),
                )
