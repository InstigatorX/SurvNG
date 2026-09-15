"""Lifecycle and aggregate quotas for opt-in encoded evidence collectors."""
from __future__ import annotations

import logging
import threading
import time

from .main_evidence import MainEvidenceProvider


LOGGER = logging.getLogger(__name__)


class MainEvidenceFleet:
    def __init__(self, config, cameras, decode_budget, enabled, *, provider_factory=MainEvidenceProvider):
        self.providers = {}
        self._enabled = enabled
        self._stop = threading.Event()
        self._thread = None
        self._retry_at = {}
        self._lock = threading.RLock()
        self._provider_factory = provider_factory
        self._decode_budget = decode_budget
        self._signatures = {}
        self.reconfigure(config, cameras)

    def reconfigure(self, config, cameras):
        with self._lock:
            self._reconfigure(config, cameras)

    def _reconfigure(self, config, cameras):
        selected = {camera.id: camera for camera in cameras if config.enabled_for(camera)}
        quota = min(config.camera_max_bytes, config.total_max_bytes // max(1, len(selected)))
        if selected and quota < 2 * 1024 * 1024:
            raise ValueError("main_evidence total quota must allow at least 2 MiB per selected camera")
        signatures = {camera_id: (camera.source_url("main"), quota, config.history_seconds,
                      config.max_timestamp_uncertainty_seconds, config.decoder)
                      for camera_id, camera in selected.items()}
        replacements = {}
        for camera_id, camera in selected.items():
            if self._signatures.get(camera_id) == signatures[camera_id]:
                replacements[camera_id] = self.providers[camera_id]
                continue
            replacements[camera_id] = self._provider_factory(
                camera_id, camera.source_url("main"), self._decode_budget,
                history_seconds=config.history_seconds, max_bytes=quota,
                max_timestamp_uncertainty_seconds=config.max_timestamp_uncertainty_seconds,
                decoder=config.decoder,
            )
        # Retire old collectors before admitting replacements under the quota.
        for camera_id, provider in self.providers.items():
            if replacements.get(camera_id) is not provider:
                provider.stop()
                self._retry_at.pop(camera_id, None)
        self.providers = replacements
        self._signatures = signatures

    def start(self):
        if not self.providers or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="main-evidence-lifecycle", daemon=True)
        self._thread.start()

    def reconcile(self):
        with self._lock:
            self._reconcile()

    def _reconcile(self):
        for camera_id, provider in self.providers.items():
            if self._stop.is_set():
                return
            try:
                enabled = self._enabled(camera_id)
                if not enabled:
                    if provider.running:
                        provider.stop()
                elif not provider.running and time.monotonic() >= self._retry_at.get(camera_id, 0):
                    self._retry_at[camera_id] = time.monotonic() + 15.0
                    provider.start()
                elif provider.running:
                    provider.refresh_status()
            except Exception:
                # Start attempts are bounded by the same reconnect backoff.
                self._retry_at[camera_id] = time.monotonic() + 15.0
                LOGGER.exception("main evidence lifecycle failed for camera %s", camera_id)

    def _run(self):
        while not self._stop.is_set():
            self.reconcile()
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("main evidence lifecycle did not stop")
        with self._lock:
            for provider in self.providers.values():
                provider.stop()

    def status(self):
        return {camera_id: provider.status() for camera_id, provider in self.providers.items()}
