"""Measurement-backed guidance for object_worker_count (Campaign 10).

Guidance only — does not change process architecture or the default of 1.
"""

from __future__ import annotations

from typing import Any, Mapping

MIN_OBJECT_WORKERS = 1
MAX_OBJECT_WORKERS = 4

# Initial admission timeout is 750ms; sustained p95 above HIGH suggests backlog.
HIGH_ADMISSION_WAIT_MS_P95 = 100.0
LOW_ADMISSION_WAIT_MS_P95 = 40.0

# Pending / queue pressure relative to the current worker pool.
PENDING_PRESSURE_PER_WORKER = 1


def clamp_object_worker_count(value: object) -> int:
    try:
        count = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        count = MIN_OBJECT_WORKERS
    return max(MIN_OBJECT_WORKERS, min(MAX_OBJECT_WORKERS, count))


def classify_device_class(device: object) -> str:
    """Classify a device string as cpu, accelerator, or auto."""
    text = str(device or "").strip().upper()
    if not text or text == "AUTO":
        return "auto"
    if text == "CPU" or text.startswith("CPU:"):
        return "cpu"
    accelerator_tokens = (
        "GPU",
        "NPU",
        "CUDA",
        "MYRIAD",
        "HDDL",
        "GNA",
        "HETERO",
        "MULTI",
        "NVIDIA",
        "INTEL.GPU",
        "OPENCL",
    )
    if any(token in text for token in accelerator_tokens):
        return "accelerator"
    # Unknown vendor strings often map to an accelerator path; stay cautious.
    return "accelerator"


def recommend_object_worker_count(
    *,
    current: int,
    device: str = "",
    initial_admission_wait_ms_p95: float | None = None,
    refinement_admission_wait_ms_p95: float | None = None,
    pending_requests: int = 0,
    queue_depth: int = 0,
    initial_waiting: int = 0,
    refinement_active: int = 0,
    security_waiting: int = 0,
    decode_waiting: int = 0,
) -> dict[str, Any]:
    """Suggest an object_worker_count in 1–4 from runtime signals.

    Heuristic:
    - Start from the current count (clamped to 1–4).
    - Low admission waits → keep current (idle single-worker stays at 1).
    - Multi-core CPU + high initial admission p95 + pending pressure → suggest
      +1, typically toward 2, and up to 4 only when pressure remains extreme.
    - Discrete GPU/NPU (or AUTO): recommend cautiously (often 1–2). If count > 1
      with high wait still, prefer 1–2 for contention — do not blindly scale to 4.
    - Never recommend above 4 or below 1.
    """
    current_count = clamp_object_worker_count(current)
    device_class = classify_device_class(device)
    pending = max(0, int(pending_requests), int(queue_depth))
    initial_wait = _finite_or_none(initial_admission_wait_ms_p95)
    refinement_wait = _finite_or_none(refinement_admission_wait_ms_p95)
    initial_waiting_count = max(0, int(initial_waiting))
    refinement_active_count = max(0, int(refinement_active))
    security_waiting_count = max(0, int(security_waiting))
    decode_waiting_count = max(0, int(decode_waiting))

    pressure_threshold = max(1, current_count * PENDING_PRESSURE_PER_WORKER)
    pending_pressure = (
        pending >= pressure_threshold
        or initial_waiting_count > 0
        or decode_waiting_count > 0
    )
    high_initial_wait = (
        initial_wait is not None and initial_wait >= HIGH_ADMISSION_WAIT_MS_P95
    )
    high_refinement_wait = (
        refinement_wait is not None and refinement_wait >= HIGH_ADMISSION_WAIT_MS_P95
    )
    low_waits = (
        (initial_wait is None or initial_wait <= LOW_ADMISSION_WAIT_MS_P95)
        and (refinement_wait is None or refinement_wait <= LOW_ADMISSION_WAIT_MS_P95)
        and pending < pressure_threshold
        and initial_waiting_count == 0
    )

    reasons: list[str] = []
    recommended = current_count

    if low_waits:
        recommended = current_count
        if current_count == 1:
            reasons.append("admission waits and queue pressure are low; keep 1 worker")
        else:
            reasons.append(
                "admission waits and queue pressure are low; keep current worker count"
            )
    elif device_class == "cpu":
        recommended, reasons = _recommend_for_cpu(
            current_count=current_count,
            high_initial_wait=high_initial_wait,
            pending_pressure=pending_pressure,
            pending=pending,
            pressure_threshold=pressure_threshold,
            high_refinement_wait=high_refinement_wait,
            refinement_active=refinement_active_count,
            initial_waiting=initial_waiting_count,
        )
    else:
        recommended, reasons = _recommend_for_accelerator(
            current_count=current_count,
            device_class=device_class,
            high_initial_wait=high_initial_wait,
            high_refinement_wait=high_refinement_wait,
            pending_pressure=pending_pressure,
            security_waiting=security_waiting_count,
        )

    if security_waiting_count > 0 and recommended > current_count:
        # Do not scale up while security work is already queued ahead.
        recommended = current_count
        reasons.append("security workload is waiting; defer scaling up")

    recommended = clamp_object_worker_count(recommended)
    if not reasons:
        reasons.append("keep current worker count")

    signals = {
        "device": str(device or ""),
        "device_class": device_class,
        "current_object_worker_count": current_count,
        "initial_admission_wait_ms_p95": initial_wait,
        "refinement_admission_wait_ms_p95": refinement_wait,
        "pending_requests": max(0, int(pending_requests)),
        "queue_depth": max(0, int(queue_depth)),
        "pending_pressure": pending_pressure,
        "initial_waiting": initial_waiting_count,
        "refinement_active": refinement_active_count,
        "security_waiting": security_waiting_count,
        "decode_waiting": decode_waiting_count,
    }
    return {
        "recommended": recommended,
        "current": current_count,
        "reasons": reasons,
        "signals": signals,
    }


def object_worker_recommendation_from_status(
    detector_status: Mapping[str, Any],
    *,
    recorded_decode: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a recommendation from detector.status() (+ optional decode budget)."""
    isolation = dict(detector_status.get("isolation") or {})
    runtime = dict(detector_status.get("runtime") or {})
    workloads = dict(runtime.get("workloads") or {})
    classes = dict(workloads.get("classes") or {})
    initial = dict(classes.get("incident_initial") or {})
    refinement = dict(classes.get("incident_refinement") or {})
    decode = dict(recorded_decode or detector_status.get("recorded_decode") or {})

    current = (
        detector_status.get("object_worker_count")
        if detector_status.get("object_worker_count") is not None
        else isolation.get("configured_workers")
    )
    device = (
        detector_status.get("loaded_device")
        or detector_status.get("configured_device")
        or isolation.get("configured_device")
        or ""
    )
    return recommend_object_worker_count(
        current=clamp_object_worker_count(current if current is not None else 1),
        device=str(device),
        initial_admission_wait_ms_p95=_finite_or_none(
            initial.get("admission_wait_ms_p95")
        ),
        refinement_admission_wait_ms_p95=_finite_or_none(
            refinement.get("admission_wait_ms_p95")
        ),
        pending_requests=int(isolation.get("pending_requests") or 0),
        queue_depth=int(runtime.get("queue_depth") or runtime.get("pending_frames") or 0),
        initial_waiting=int(workloads.get("initial_waiting") or 0),
        refinement_active=int(workloads.get("refinement_active") or 0),
        security_waiting=int(workloads.get("security_waiting") or 0),
        decode_waiting=int(decode.get("waiting") or 0),
    )


def _recommend_for_cpu(
    *,
    current_count: int,
    high_initial_wait: bool,
    pending_pressure: bool,
    pending: int,
    pressure_threshold: int,
    high_refinement_wait: bool,
    refinement_active: int,
    initial_waiting: int,
) -> tuple[int, list[str]]:
    if not (high_initial_wait and pending_pressure):
        return current_count, [
            "CPU device without combined high initial wait and pending pressure; "
            "keep current worker count"
        ]

    extreme = (
        pending >= pressure_threshold * 2
        or (high_refinement_wait and refinement_active > 0)
        or initial_waiting >= max(2, current_count)
    )
    if current_count < 2:
        return 2, [
            "CPU device with high initial admission wait and pending pressure; "
            "suggest 2 workers"
        ]
    if extreme and current_count < MAX_OBJECT_WORKERS:
        recommended = min(MAX_OBJECT_WORKERS, current_count + 1)
        return recommended, [
            "CPU device still under extreme admission/queue pressure; "
            f"suggest {recommended} workers"
        ]
    return current_count, [
        f"CPU device shows elevated waits; keep current count at {current_count}"
    ]


def _recommend_for_accelerator(
    *,
    current_count: int,
    device_class: str,
    high_initial_wait: bool,
    high_refinement_wait: bool,
    pending_pressure: bool,
    security_waiting: int,
) -> tuple[int, list[str]]:
    label = "accelerator" if device_class == "accelerator" else "AUTO"
    elevated = high_initial_wait or high_refinement_wait or pending_pressure

    if not elevated:
        recommended = min(current_count, 2)
        return recommended, [
            f"{label} device without strong backlog pressure; "
            f"recommend {recommended} (cap 2)"
        ]

    # Contention on a single discrete device: prefer 1–2, never jump to 4.
    if current_count > 2:
        return 2, [
            f"{label} device with sustained waits at count>2; "
            "prefer 1–2 workers to reduce device contention"
        ]
    if current_count == 1 and pending_pressure and high_initial_wait:
        return 2, [
            f"{label} device with high initial wait and pending pressure; "
            "cautiously suggest 2 workers"
        ]
    if (
        current_count >= 2
        and high_initial_wait
        and not pending_pressure
        and security_waiting > 0
    ):
        return 1, [
            f"{label} device contention with security waiting; prefer 1 worker"
        ]
    recommended = min(2, max(1, current_count))
    return recommended, [
        f"{label} device shows elevated waits; keep 1–2 workers ({recommended})"
    ]


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number
