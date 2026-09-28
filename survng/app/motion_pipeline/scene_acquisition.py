"""Acquire first; incident establishment is a separate, replayable decision.

This adapter owns no grouping policy. Both discovery and bounded confirmation
persist their actual sampled frames before invoking the activity evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

from ..scene_zone_admission import discovery_requires_confirmation, evaluate_scene_establishment
from ..detector import detection_failure
from ..scene_identity import analysis_observation_ids


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:32]


def acquire_detection_result(events, camera_id, event_at, qualification, frame, objects,
                             provider_result, snapshot_writer):
    """Return persisted evidence and a decision, never an implicit incident.

    The caller may admit a supported decision in the event transaction. A crash
    between these operations leaves a replayable decision, not a partial scene.
    """
    at = event_at.timestamp()
    batches = [item for item in objects if item.get("status") == "scene_observations"]
    samples = [sample for batch in batches for sample in batch.get("samples", [])]
    captured = getattr(provider_result, "frame_captured_at_epoch", None)
    captured = float(captured) if isinstance(captured, (int, float)) and math.isfinite(captured) else at
    source = str(getattr(provider_result, "frame_source", "") or "unknown")
    observations = [o for batch in batches for o in batch.get("observations", [])]
    if not samples:
        samples = [{"camera_id": camera_id, "captured_epoch": captured,
                    "status": "complete" if frame is not None and not detection_failure(objects) else "failed",
                    "observations": observations,
                    "metadata": {"source": source,
                                 "confirmation_complete": bool(qualification.get("scene_confirmation")),
                                 "frame_width": int(frame.shape[1]) if frame is not None else 0,
                                 "frame_height": int(frame.shape[0]) if frame is not None else 0}}]
    # A review frame is its own evidence. It never replaces a seed acquisition's
    # image, timestamps, coordinate plane, or raw detector payload.
    snapshot_path = ""
    persisted = []
    for sample in samples:
        sample_at = float(sample["captured_epoch"])
        metadata = dict(sample.get("metadata") or {})
        sample_source = str(metadata.get("source") or sample.get("source") or source)
        metadata.setdefault("source", sample_source)
        status = sample.get("status", "complete")
        raw = [dict(o.get("payload", o)) for o in sample.get("observations", [])]
        sample_id = sample.get("id") or "sample-" + _key([camera_id, sample_at, sample_source,
                      metadata.get("camera_generation"), metadata.get("capture_generation"),
                      analysis_observation_ids(camera_id, sample_at, raw)])
        if status == "failed":
            # A failed attempt is not the identity of a later successful frame.
            sample_id += "-failed-" + str(qualification.get("scene_candidate_lease_token", 0))
        is_review_frame = abs(sample_at - captured) <= .05 and sample_source == source
        if is_review_frame and frame is not None and (observations or qualification.get("scene_confirmation")):
            existing = events.scene_sample(sample_id)
            retained = existing.get("snapshot_path") if existing else ""
            snapshot_path = str(events._snapshot_path_for_retention(retained)) if retained else snapshot_writer(frame, datetime.fromtimestamp(captured, timezone.utc))
        path = snapshot_path if is_review_frame else ""
        if path:
            metadata["snapshot_size_bytes"] = events._snapshot_file_size(path)
            for observation in raw:
                observation["snapshot_path"] = path
        metadata["image"] = {"width": metadata.get("frame_width", 0),
                             "height": metadata.get("frame_height", 0), "source": sample_source,
                             "captured_at": datetime.fromtimestamp(sample_at, timezone.utc).isoformat(),
                             "analyzed_frame": True}
        request = None
        confirm = status == "failed" or discovery_requires_confirmation(
            raw, qualification.get("establishment_zone_policy"),
            frame_width=int(metadata.get("frame_width") or 0),
            frame_height=int(metadata.get("frame_height") or 0),
        )
        if (raw or status == "failed") and confirm and not qualification.get("scene_confirmation"):
            request = {"start_epoch": sample_at - 10, "end_epoch": sample_at + 5,
                       "deadline_epoch": time.time() + 300, "available_at_epoch": time.time() + 15}
        acquired = events.acquire_scene_sample(
            sample_id=sample_id, camera_id=camera_id, captured_epoch=sample_at,
            source=sample_source, status=status, observations=raw, metadata=metadata,
            snapshot_path=path, recording_path=str(sample.get("recording_path") or metadata.get("recording_path") or ""),
            request_confirmation=request,
        )
        persisted.append(events.scene_sample(acquired["id"]))
    policy = qualification.get("establishment_zone_policy")
    assessment = evaluate_scene_establishment(persisted, policy=policy if isinstance(policy, dict) else None)
    ids = [sample["id"] for sample in persisted]
    seed_ids = list(qualification.get("scene_seed_sample_ids") or [])
    all_ids = list(dict.fromkeys([*seed_ids, *ids]))
    decision_id = "activity-" + _key([all_ids, assessment])
    events.record_scene_activity_decision(
        decision_id=decision_id, sample_ids=all_ids, verdict=assessment["status"],
        activity_epoch=assessment.get("activity_epoch"), reason=assessment["reason"],
        policy_version=assessment.get("policy_version", 1), evidence=assessment,
    )
    qualification.update(scene_sample_ids=all_ids, scene_activity_decision_id=decision_id)
    associate = getattr(events, "associate_scene_samples", None)
    updates = associate(all_ids) if callable(associate) else []
    return {"decision_id": decision_id, "assessment": assessment,
            "sample_ids": all_ids, "snapshot_path": snapshot_path, "context_updates": updates}
