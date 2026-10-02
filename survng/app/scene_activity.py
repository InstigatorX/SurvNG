"""Pure, replayable activity decisions over persisted acquisition evidence.

Object confidence, detector label changes and alert admission are deliberately
not activity witnesses. Pixel witnesses are produced while source frames are
available; this evaluator verifies their references against the durable ledger.
"""
from __future__ import annotations

import math
from typing import Any


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def evaluate_scene_activity(samples, *, policy_version=1, ignore_stationary_scene_context=False) -> dict[str, Any]:
    if policy_version != 1:
        raise ValueError("unsupported scene activity policy version")
    # A retry, a second model, or a second resolution of one capture does not
    # constitute another instant of physical evidence.
    complete = {}
    by_id = {}
    failed = 0
    finished = False
    for raw in samples:
        if not isinstance(raw, dict):
            continue
        captured = _number(raw.get("captured_epoch"))
        metadata = raw.get("metadata") or {}
        finished = finished or metadata.get("confirmation_complete") is True
        if raw.get("status") != "complete" or captured is None:
            failed += 1
            continue
        camera = str(raw.get("camera_id") or "")
        key = (camera, captured)
        complete.setdefault(key, []).append(raw)
        if raw.get("id"):
            by_id[str(raw["id"])] = raw
    epochs = [key[1] for key in complete]
    summary = {"policy_version": policy_version, "sample_count": len(complete)+failed,
               "complete_sample_count": len(complete), "failed_sample_count": failed,
               "witness_count": 0, "evidence_kind": "none"}
    result = {"status": "pending", "reason": "awaiting_activity_confirmation",
              "summary": "Waiting for evidence of physical activity.", "diagnostics": summary,
              "supporting_observation_ids": [], "activity_epoch": None,
              "start_epoch": min(epochs) if epochs else None}
    supported = []
    witnesses = []
    for variants in complete.values():
        for sample in variants:
            metadata = sample.get("metadata") or {}
            if metadata.get("validated_motion") is True:
                if metadata.get("motion_source") != "camera_reported":
                    supported.append((float(sample["captured_epoch"]), [], "measured_motion"))
                    witnesses.append({"epoch": float(sample["captured_epoch"]), "observation_ids": [],
                                      "kind": "measured_motion"})
            observations = {}
            for observation in sample.get("observations", []):
                if not isinstance(observation, dict):
                    continue
                payload = observation.get("payload", observation)
                identifier = observation.get("id", payload.get("id", payload.get("observation_key")))
                if identifier:
                    observations[str(identifier)] = payload
            for witness in metadata.get("activity_witnesses", []):
                if not isinstance(witness, dict) or witness.get("validated") is not True:
                    continue
                if witness.get("kind") not in {"localized_arrival", "localized_movement", "localized_motion"}:
                    continue
                previous = by_id.get(str(witness.get("from_sample_id")))
                if previous is None or str(previous.get("camera_id") or "") != str(sample.get("camera_id") or ""):
                    continue
                before, after = _number(previous.get("captured_epoch")), _number(sample.get("captured_epoch"))
                if before is None or after is None or not .05 <= after-before <= 10:
                    continue
                if witness.get("to_sample_id") != sample.get("id"):
                    continue
                if witness.get("camera_stable") is not True:
                    continue
                local = _number(witness.get("local_change_fraction"))
                background = _number(witness.get("background_change_fraction"))
                if local is None or background is None or local < .08 or background > .08:
                    continue
                ids = [str(key) for key in witness.get("observation_ids", []) if str(key) in observations]
                if not ids:
                    continue
                if witness["kind"] == "localized_movement":
                    displacement = _number(witness.get("normalized_displacement"))
                    if displacement is None or displacement < .01:
                        continue
                if (
                    ignore_stationary_scene_context
                    and witness["kind"] == "localized_motion"
                    and all(str(observations[key].get("activity_role") or "") == "scene_context" for key in ids)
                ):
                    # In-place pixel change on a known stationary subject is
                    # retained on the sample, but it is not activity.
                    continue
                supported.append((after, ids, "video_verified_activity"))
                witnesses.append({"epoch": after, "observation_ids": ids, "kind": "video_verified_activity"})
    result["witnesses"] = witnesses
    if supported:
        summary.update(witness_count=len(supported), evidence_kind=(
            "video_verified_activity" if any(s[2] == "video_verified_activity" for s in supported)
            else supported[0][2]))
        result.update(status="supported", reason=summary["evidence_kind"],
                      summary="Physical scene activity is supported by recorded evidence.",
                      activity_epoch=max(s[0] for s in supported),
                      supporting_observation_ids=sorted({key for _, ids, _ in supported for key in ids}))
    elif failed or not complete:
        result.update(status="incomplete", reason="source_evidence_unavailable",
                      summary="Source evidence is unavailable; activity remains unresolved.")
    elif finished:
        result.update(status="unsupported" if len(complete) >= 2 else "incomplete",
                      reason="no_supported_physical_change" if len(complete) >= 2 else "insufficient_distinct_frames",
                      summary="No physical activity was established in the sampled interval." if len(complete)>=2
                      else "There are too few distinct frames to assess activity.")
    return result
