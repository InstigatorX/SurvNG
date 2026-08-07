"""Transactional replacement of an application-manager generation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig
from .manager import AppManager, ManagerShutdownIncompleteError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManagerReloadHooks:
    active_storage_tasks: Callable[[AppManager], list[str]]
    active_ai_operations: Callable[[], dict[str, int]]
    prewarmer_running: Callable[[], bool]
    stop_prewarmer: Callable[[], None]
    start_prewarmer: Callable[[], None]
    save_config: Callable[..., None]
    publish_runtime: Callable[[AppConfig, AppManager], None]
    refresh_runtime_caches: Callable[[], None]
    storage_error: Callable[[list[str]], Exception]
    ai_error: Callable[[dict[str, int]], Exception]


class ManagerGenerationLifecycle:
    """Own full-manager cutover, recovery, and publication as one transaction."""

    def __init__(
        self,
        *,
        lock: threading.RLock,
        stopping: threading.Event,
        manager_factory: Callable[[AppConfig], AppManager],
        hooks: ManagerReloadHooks,
    ) -> None:
        self._lock = lock
        self._stopping = stopping
        self._manager_factory = manager_factory
        self._hooks = hooks

    def reload(
        self,
        previous_config: AppConfig,
        previous_manager: AppManager,
        effective_config: AppConfig,
        *,
        persist: bool,
    ) -> None:
        with self._lock:
            if self._stopping.is_set():
                raise RuntimeError(
                    "configuration reload refused while SurvNG is shutting down"
                )
            if tasks := self._hooks.active_storage_tasks(previous_manager):
                raise self._hooks.storage_error(tasks)
            if operations := self._hooks.active_ai_operations():
                raise self._hooks.ai_error(operations)

            prewarmer_was_running = self._hooks.prewarmer_running()
            candidate = self._manager_factory(effective_config)
            previous_stop_attempted = False
            try:
                preferences = previous_manager.runtime_preferences()
                self._hooks.stop_prewarmer()
                previous_stop_attempted = True
                previous_manager.stop_all_with_runtime_preferences()
                candidate.apply_runtime_preferences(preferences)
                candidate.start_all()
                candidate.apply_runtime_preferences(preferences, persist=True)
                if persist:
                    self._hooks.save_config(effective_config, assign_ids=False)
            except BaseException as reload_error:
                try:
                    candidate.stop_all()
                except Exception:
                    LOGGER.exception(
                        "failed to clean up replacement manager after reload failure"
                    )
                if isinstance(reload_error, ManagerShutdownIncompleteError):
                    self._hooks.publish_runtime(previous_config, previous_manager)
                    raise RuntimeError(
                        "configuration reload aborted because camera shutdown is "
                        "still active; restart SurvNG through its supervisor"
                    ) from None
                if not previous_stop_attempted:
                    if prewarmer_was_running:
                        self._hooks.start_prewarmer()
                    raise RuntimeError(
                        "configuration reload failed before the active manager was stopped"
                    ) from reload_error
                try:
                    recovery = self._manager_factory(previous_config)
                    recovery.apply_runtime_preferences(preferences, persist=True)
                    recovery.start_all()
                except BaseException as recovery_error:
                    raise RuntimeError(
                        "configuration reload failed and the previous manager could not be restored"
                    ) from recovery_error
                self._hooks.publish_runtime(previous_config, recovery)
                self._hooks.refresh_runtime_caches()
                if prewarmer_was_running:
                    self._hooks.start_prewarmer()
                if not isinstance(reload_error, Exception):
                    raise
                raise RuntimeError(
                    "configuration reload failed; the previous configuration was restored"
                ) from reload_error

            self._hooks.publish_runtime(effective_config, candidate)
            self._hooks.refresh_runtime_caches()
            if prewarmer_was_running:
                self._hooks.start_prewarmer()
