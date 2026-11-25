"""Transactional replacement of an application-manager generation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig
from .manager import AppManager

LOGGER = logging.getLogger(__name__)


class ManagerReloadRestartPending(RuntimeError):
    """The reload cannot finish in this process; the supervisor restarts it.

    ``persisted`` is true when the requested configuration is on disk, so the
    restarted process starts with it; ``restarting`` is true when the
    supervisor accepted the restart (otherwise someone must restart SurvNG).
    """

    def __init__(self, message: str, *, persisted: bool, restarting: bool = True) -> None:
        super().__init__(message)
        self.persisted = persisted
        self.restarting = restarting


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
    wait_for_manager_idle: Callable[[AppManager], bool] = lambda _manager: True
    # Asked to stop the process when this generation cannot recover in-process;
    # returns True when the supervisor will restart SurvNG.
    request_supervisor_restart: Callable[[str], bool] = lambda _reason: False
    # Records a change in configuration history once it takes effect (now, or
    # at the restart that applies it). save_config itself records nothing,
    # so a change that is saved and then rolled back leaves no history.
    record_change: Callable[[AppConfig, AppConfig], None] = lambda _previous, _current: None


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

    def _unrecoverable(
        self,
        problem: str,
        *,
        persisted: bool = False,
    ) -> ManagerReloadRestartPending:
        """Ask the supervisor to restart SurvNG and describe the failure.

        Called after the stopping flag is set and the surviving generation is
        published, so the final shutdown cleans up the right manager.
        """
        try:
            restarting = bool(self._hooks.request_supervisor_restart(problem))
        except Exception:
            LOGGER.exception("could not request a supervisor restart")
            restarting = False
        action = (
            "SurvNG is restarting through its supervisor"
            if restarting
            else "restart SurvNG through its supervisor"
        )
        if persisted:
            action += " to apply the saved configuration"
        return ManagerReloadRestartPending(
            f"{problem}; {action}",
            persisted=persisted,
            restarting=restarting,
        )

    def _record_change(self, previous_config: AppConfig, effective_config: AppConfig) -> None:
        try:
            self._hooks.record_change(previous_config, effective_config)
        except Exception:
            LOGGER.exception("could not record the configuration change")

    def _restore_saved(self, previous_config: AppConfig) -> None:
        """Put the configuration that is running back on disk."""
        try:
            self._hooks.save_config(previous_config, assign_ids=False)
        except Exception:
            LOGGER.exception(
                "failed to restore the persisted configuration after a reload failure"
            )

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
            previous_stopped = False
            persisted = False
            try:
                preferences = previous_manager.runtime_preferences()
                self._hooks.stop_prewarmer()
                if not self._hooks.wait_for_manager_idle(previous_manager):
                    raise RuntimeError(
                        "configuration reload timed out waiting for active API requests"
                    )
                # Persist the request before anything irreversible: when this
                # process cannot finish the cutover, its restart must start
                # with what was asked for rather than silently drop it.
                if persist:
                    self._hooks.save_config(effective_config, assign_ids=False)
                    persisted = True
                previous_stop_attempted = True
                previous_manager.stop_all_with_runtime_preferences()
                previous_stopped = True
                candidate.apply_runtime_preferences(preferences)
                candidate.start_all()
                candidate.apply_runtime_preferences(preferences, persist=True)
            except BaseException as reload_error:
                cleanup_error = None
                try:
                    candidate.stop_all()
                except Exception as error:
                    cleanup_error = error
                    LOGGER.exception(
                        "failed to clean up replacement manager after reload failure"
                    )
                if previous_stop_attempted and not previous_stopped:
                    # The new configuration was never tried; the old
                    # generation would not stop. Keep the request on disk so
                    # the restart applies it.
                    self._stopping.set()
                    self._hooks.publish_runtime(previous_config, previous_manager)
                    if persisted:
                        self._record_change(previous_config, effective_config)
                    raise self._unrecoverable(
                        "configuration reload aborted because previous manager shutdown is "
                        "still active",
                        persisted=persisted,
                    ) from None
                if persisted:
                    # The new configuration was tried and failed, or the
                    # cutover never began: what runs is the previous one.
                    self._restore_saved(previous_config)
                if cleanup_error is not None:
                    self._stopping.set()
                    if previous_stopped:
                        # Keep the surviving generation reachable by final
                        # shutdown; never open the same cameras a second time.
                        self._hooks.publish_runtime(effective_config, candidate)
                    raise self._unrecoverable(
                        "replacement manager cleanup is incomplete"
                    ) from cleanup_error
                if not previous_stop_attempted:
                    if prewarmer_was_running:
                        self._hooks.start_prewarmer()
                    raise RuntimeError(
                        "configuration reload failed before the active manager was stopped"
                    ) from reload_error
                recovery = None
                try:
                    recovery = self._manager_factory(previous_config)
                    recovery.apply_runtime_preferences(preferences, persist=True)
                    recovery.start_all()
                except BaseException as recovery_error:
                    self._stopping.set()
                    if recovery is not None:
                        try:
                            recovery.stop_all()
                        except Exception:
                            LOGGER.exception("failed to clean up recovery manager")
                            self._hooks.publish_runtime(previous_config, recovery)
                    raise self._unrecoverable(
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

            if persisted:
                self._record_change(previous_config, effective_config)
            self._hooks.publish_runtime(effective_config, candidate)
            self._hooks.refresh_runtime_caches()
            if prewarmer_was_running:
                self._hooks.start_prewarmer()
