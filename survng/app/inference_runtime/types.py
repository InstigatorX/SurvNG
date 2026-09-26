from __future__ import annotations

from enum import IntEnum
import logging


LOGGER = logging.getLogger("uvicorn.error")
MAX_INFERENCE_FRAME_BYTES = 64 * 1024 * 1024
INFERENCE_START_TIMEOUT_SECONDS = 30.0
INFERENCE_REQUEST_TIMEOUT_SECONDS = 15.0
INCIDENT_INITIAL_WORKER_TIMEOUT_SECONDS = 3.0
INCIDENT_INITIAL_ADMISSION_TIMEOUT_SECONDS = 0.75
PERSON_REID_REQUEST_TIMEOUT_SECONDS = 3.0
INFERENCE_STATUS_TIMEOUT_SECONDS = 5.0
INFERENCE_RESTART_DELAY_SECONDS = 1.0
INFERENCE_CRASH_WINDOW_SECONDS = 10 * 60.0
INFERENCE_GPU_FALLBACK_CRASHES = 3
INFERENCE_GPU_FALLBACK_SECONDS = 30 * 60.0
RESOURCE_TRACKER_STOP_TIMEOUT_SECONDS = 2.0


class InferenceUnavailable(RuntimeError):
    pass


class InferenceRollbackIncomplete(InferenceUnavailable):
    """A failed reconfiguration could not restore the previous safe pool."""


class InferenceWorkload(IntEnum):
    """Security work runs before optional enrichment on each worker."""

    INCIDENT_INITIAL = 0
    INCIDENT_REFINEMENT = 1
    INTERACTIVE = 2
    TRACKING = 3
    ENRICHMENT = 4
    OFFLINE = 5

    # Compatibility for callers/tests that used the original coarse class.
    INCIDENT = INCIDENT_INITIAL

