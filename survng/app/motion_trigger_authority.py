"""Additive source provenance shared by delivery and its retry checkpoints."""

from __future__ import annotations

from typing import Any

from .durable_payload import durable_json_copy, durable_json_dumps


def merge_trigger_authority(
    payload: dict[str, Any], authority: dict[str, Any]
) -> dict[str, Any]:
    """Preserve admitted sources when an older consumer checkpoints its work.

    Camera reports are advisory and bounded, as they are at ingress. Neither
    source merging nor report retention changes the trigger's occurrence,
    evidence-frame token, or current camera configuration.
    """
    merged = dict(payload)
    merged["admitted_sources"] = sorted({
        source
        for value in (payload, authority)
        for source in (value.get("admitted_sources") or ())
        if source in {"camera", "manual", "ema"}
    })
    reports: dict[str, dict[str, Any]] = {}
    has_semantic_snapshot = False
    for value in (payload, authority):
        semantics = value.get("camera_semantics")
        if not isinstance(semantics, dict):
            continue
        has_semantic_snapshot = True
        for report in semantics.get("reports", ()):
            if not isinstance(report, dict):
                continue
            key = durable_json_dumps(report, sort_keys=True)
            reports[key] = report
    if has_semantic_snapshot:
        # Canonical ordering also makes retries and restored controllers expose
        # exactly the same bounded snapshot, regardless of merge ordering.
        # Empty parsed snapshots must survive too: None means legacy/unparsed.
        merged["camera_semantics"] = {"reports": [
            durable_json_copy(reports[key]) for key in sorted(reports)[:64]
        ]}
    return merged
