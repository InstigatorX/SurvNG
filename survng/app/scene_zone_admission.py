"""Zone interpretation for incident establishment.

Physical activity and alert eligibility stay independent. A zone can keep
activity from opening or prolonging an incident, and it never removes an
observation from an incident that eligible activity already established.
Notification settings are not part of this policy. The detector acceptance
threshold is snapshotted separately so a located box still has to meet it.
"""
from __future__ import annotations

import math
from typing import Any

from .config import CameraConfig, DetectionZone
from .motion_correlation import assess_motion_objects
from .scene_activity import _number, evaluate_scene_activity
from .scene_context_memory import normalized_box
from .stationary_policy import stationary_object_policy
from .zones import apply_depth_zone_filters, apply_detection_zones, class_confidence_threshold

ESTABLISHMENT_ZONE_POLICY_VERSION = "establishment_zones_v1"
ESTABLISHMENT_ZONE_POLICY_VERSION_V2 = "establishment_zones_v2"
# A box verifies where a notice occurred. Only boxes this soon after the notice
# date it; a stationary subject seen later is not further activity.
NOTICE_VERIFICATION_SECONDS = 10.0


def effective_stationary_presence(camera: CameraConfig, detector_config: Any | None = None) -> str:
    """Resolve a camera inherit to the detector's presence mode."""
    value = getattr(camera, "stationary_subject_presence", "inherit")
    if value == "inherit":
        value = getattr(detector_config, "stationary_subject_presence", "ignore")
    return value if value in {"ignore", "activity"} else "ignore"


def effective_activity_attribution(camera: CameraConfig, detector_config: Any | None = None) -> str:
    value = getattr(camera, "object_activity_attribution", "inherit")
    if value == "inherit":
        value = getattr(detector_config, "object_activity_attribution", "enforce")
    return value if value in {"off", "shadow", "enforce"} else "enforce"


def stationary_presence_enforced(policy: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(policy, dict)
        and policy.get("stationary_subject_presence") == "ignore"
        and policy.get("object_activity_attribution") == "enforce"
    )


def stationary_presence_shadow(policy: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(policy, dict)
        and policy.get("stationary_subject_presence") == "ignore"
        and policy.get("object_activity_attribution") == "shadow"
    )


def establishment_zone_policy(
    camera: CameraConfig,
    *,
    require_incident_zone: bool | None = None,
    confidence_threshold: float | None = None,
    class_confidence_thresholds: dict[str, float] | None = None,
    detector_config: Any | None = None,
) -> dict[str, Any]:
    """Snapshot the geometry used to admit activity, without alert settings.

    ``notifications_enabled`` and per-zone confidence thresholds are omitted
    on purpose. The detector acceptance threshold is included when the caller
    has one, so a later replay does not borrow whatever threshold is configured
    then. A snapshot recorded without that field keeps its original meaning.
    """
    if require_incident_zone is None:
        require_incident_zone = True if camera.require_incident_zone is None else bool(camera.require_incident_zone)
    zones = []
    for zone in camera.zones:
        zones.append({
            "name": zone.name,
            "enabled": bool(zone.enabled),
            "behavior": zone.behavior,
            "points": [{"x": float(point.x), "y": float(point.y)} for point in zone.points],
            "object_classes": list(zone.object_classes),
            "min_depth_m": zone.min_depth_m,
            "max_depth_m": zone.max_depth_m,
        })
    policy = {
        "version": 1,
        "require_incident_zone": bool(require_incident_zone),
        "zones": zones,
    }
    if confidence_threshold is not None:
        policy["confidence_threshold"] = float(confidence_threshold)
    normalized = _class_thresholds(class_confidence_thresholds)
    if normalized:
        policy["class_confidence_thresholds"] = normalized
    policy["stationary_subject_presence"] = effective_stationary_presence(camera, detector_config)
    policy["object_activity_attribution"] = effective_activity_attribution(camera, detector_config)
    return policy


def _camera_from_policy(policy: dict[str, Any]) -> CameraConfig:
    zones = []
    for raw in policy.get("zones") or []:
        if not isinstance(raw, dict):
            continue
        points = raw.get("points") or []
        zones.append(DetectionZone(
            name=str(raw.get("name") or "zone"),
            enabled=bool(raw.get("enabled", True)),
            behavior=raw.get("behavior") if raw.get("behavior") in {"incident", "ignore", "none"} else "none",
            points=[{"x": float(point["x"]), "y": float(point["y"])} for point in points if isinstance(point, dict) and "x" in point and "y" in point],
            object_classes=[str(item) for item in raw.get("object_classes") or []],
            min_depth_m=raw.get("min_depth_m"),
            max_depth_m=raw.get("max_depth_m"),
        ))
    return CameraConfig(
        id="establishment-policy",
        name="establishment-policy",
        stream_url="rtsp://establishment-policy.invalid/main",
        require_incident_zone=bool(policy.get("require_incident_zone", True)),
        zones=zones,
    )


def _class_thresholds(values: dict[str, float] | None) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    normalized = {}
    for key, value in values.items():
        try:
            normalized[str(key).strip().lower()] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _admission_confidence(payload: dict[str, Any]) -> float:
    """Real score for admission. A same-track median replaces the frame score."""
    median = payload.get("semantic_median_confidence")
    try:
        if median is not None and math.isfinite(float(median)):
            return float(median)
    except (TypeError, ValueError):
        pass
    try:
        value = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _confidence_admitted(payload: dict[str, Any], policy: dict[str, Any] | None) -> bool:
    """True when this snapshot does not carry a detector threshold, or the score meets it."""
    if not isinstance(policy, dict) or "confidence_threshold" not in policy:
        return True
    try:
        default = float(policy["confidence_threshold"])
    except (TypeError, ValueError):
        return True
    if not math.isfinite(default):
        return True
    label = str(payload.get("label") or "")
    required = class_confidence_threshold(label, default, _class_thresholds(policy.get("class_confidence_thresholds")))
    return _admission_confidence(payload) >= required


def _admits_activity(item: dict[str, Any]) -> bool:
    if item.get("admission_reason") == "stationary_scene_context":
        return False
    return bool(item.get("establishment_eligible")) and item.get("confidence_admitted") is not False


def discovery_requires_confirmation(observations, policy: dict[str, Any] | None, *, frame_width: int = 0, frame_height: int = 0) -> bool:
    """Recorded confirmation is for activity that can still establish an incident.

    The discovery observations themselves stay in the acquisition ledger either
    way. A restricting snapshot with only Ignore-zone or outside-zone boxes does
    not schedule another recorded pass. The next discovery that sees an eligible
    object still includes the preceding discovery instant.
    """
    if not isinstance(policy, dict) or not policy_restricts(policy):
        return True
    if frame_width <= 0 or frame_height <= 0:
        return True
    saw_box = False
    for observation in observations or []:
        if not isinstance(observation, dict):
            continue
        saw_box = True
        if interpret_establishment_observation(
            observation, policy, frame_width=frame_width, frame_height=frame_height,
        )["establishment_eligible"]:
            return True
    return not saw_box


def policy_restricts(policy: dict[str, Any] | None) -> bool:
    """True when some of the frame cannot establish an incident."""
    if not isinstance(policy, dict):
        return False
    zones = [
        zone for zone in policy.get("zones") or []
        if isinstance(zone, dict) and zone.get("enabled", True) and zone.get("behavior") in {"incident", "ignore"}
        and len(zone.get("points") or []) >= 3
    ]
    if any(zone.get("behavior") == "ignore" for zone in zones):
        return True
    return bool(policy.get("require_incident_zone")) and any(zone.get("behavior") == "incident" for zone in zones)


def _frame_size(observation: dict[str, Any], sample: dict[str, Any]) -> tuple[int, int]:
    metadata = sample.get("metadata") or {}
    width = observation.get("detection_frame_width", metadata.get("frame_width", 0))
    height = observation.get("detection_frame_height", metadata.get("frame_height", 0))
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return 0, 0


def interpret_establishment_observation(observation: dict[str, Any], policy: dict[str, Any] | None, *, frame_width: int = 0, frame_height: int = 0) -> dict[str, Any]:
    """Spatial zone result for one box. The zone test ignores confidence.

    ``establishment_eligible`` stays spatial so a weak in-zone box can still be
    confirmed. ``confidence_admitted`` is whether that box may open or prolong
    an incident.
    """
    payload = observation.get("payload", observation) if isinstance(observation, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    identifier = str(observation.get("id") or payload.get("id") or payload.get("observation_key") or "")
    width = frame_width or _frame_size(payload, {})[0]
    height = frame_height or _frame_size(payload, {})[1]
    box = payload.get("box")
    label = str(payload.get("label") or "")
    base = {"observation_id": identifier, "establishment_eligible": False, "reason": "insufficient_spatial_evidence",
            "zones": [], "behaviors": []}
    if not isinstance(policy, dict) or not label or not isinstance(box, dict) or width <= 0 or height <= 0:
        return base
    try:
        camera = _camera_from_policy(policy)
    except (TypeError, ValueError):
        return base
    item = {"label": label, "confidence": 0.99, "box": box}
    if isinstance(payload.get("depth_stats"), dict):
        item["depth_stats"] = payload["depth_stats"]
    apply_detection_zones(camera, [item], width, height, 0.01)
    apply_depth_zone_filters(camera, [item])
    matches = item.get("spatial_zone_matches") or []
    names = [str(zone.get("name")) for zone in matches if isinstance(zone, dict) and zone.get("name")]
    behaviors = [str(zone.get("behavior")) for zone in matches if isinstance(zone, dict)]
    if item.get("depth_zone_filtered") is True:
        reason = "depth_ignore_zone"
        eligible = False
    elif item.get("spatial_zone_eligible") is True:
        reason = "incident_zone" if "incident" in behaviors else "full_frame"
        eligible = True
    elif "ignore" in behaviors:
        reason = "ignored_zone"
        eligible = False
    else:
        reason = "outside_incident_zone" if policy_restricts(policy) else "full_frame"
        eligible = reason == "full_frame"
    # The substituted confidence answers only "which zone contains the foot."
    # Opening an incident uses the detection's own score, or the median already
    # computed for this same track.
    admitted = _confidence_admitted(payload, policy)
    admission_reason = "below_confidence" if eligible and not admitted else reason
    role = str(payload.get("activity_role") or "")
    effect = ""
    if eligible and role == "scene_context":
        if stationary_presence_enforced(policy):
            effect = "enforce"
            admission_reason = "stationary_scene_context"
        elif stationary_presence_shadow(policy):
            effect = "shadow"
    return {
        "observation_id": identifier,
        "establishment_eligible": eligible,
        "reason": reason,
        "admission_reason": admission_reason,
        "confidence_admitted": admitted,
        "zones": names,
        "behaviors": behaviors,
        "activity_role": role or "indeterminate",
        "stationary_presence": policy.get("stationary_subject_presence") if isinstance(policy, dict) else None,
        "stationary_counterfactual": effect == "shadow",
    }


def _iter_observations(samples):
    for sample in samples or []:
        if not isinstance(sample, dict):
            continue
        for observation in sample.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload", observation)
            if not isinstance(payload, dict):
                continue
            width, height = _frame_size(payload, sample)
            identifier = str(observation.get("id") or payload.get("id") or payload.get("observation_key") or "")
            yield identifier, observation, width, height


def _compact_activity(activity: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": activity.get("status"),
        "reason": activity.get("reason"),
        "activity_epoch": activity.get("activity_epoch"),
        "supporting_observation_ids": list(activity.get("supporting_observation_ids") or []),
        "evidence_kind": (activity.get("diagnostics") or {}).get("evidence_kind"),
        "witnesses": [
            {"epoch": witness.get("epoch"), "observation_ids": list(witness.get("observation_ids") or []), "kind": witness.get("kind")}
            for witness in activity.get("witnesses") or []
        ],
    }


def _rejection_reason(interpretations: list[dict[str, Any]]) -> str:
    localized = [item for item in interpretations if (item.get("admission_reason") or item.get("reason")) != "insufficient_spatial_evidence"]
    if not localized:
        return "insufficient_spatial_evidence"
    reasons = {item.get("admission_reason") or item["reason"] for item in localized}
    if reasons == {"stationary_scene_context"}:
        return "stationary_scene_context"
    reasons.discard("stationary_scene_context")
    if not reasons:
        return "stationary_scene_context"
    if reasons <= {"ignored_zone", "depth_ignore_zone"}:
        return "depth_ignore_zone" if reasons == {"depth_ignore_zone"} else "ignored_zone"
    if reasons == {"outside_incident_zone"}:
        return "outside_incident_zone"
    if reasons == {"below_confidence"}:
        return "below_confidence"
    # Ignore-zone witnesses and witnesses outside every incident zone are both
    # ineligible. The headline must not claim that all of the activity was ignored.
    if "below_confidence" in reasons or ("outside_incident_zone" in reasons and ({"ignored_zone", "depth_ignore_zone"} & reasons)):
        return "ineligible_zone"
    if "ignored_zone" in reasons or "depth_ignore_zone" in reasons:
        return "ignored_zone"
    return "outside_incident_zone"


def _zone_summary(reason: str, *, established: bool, notice: bool) -> str:
    if established and notice:
        return "A camera or motion notice was verified by a detection in an eligible zone."
    if established:
        return "Physical activity in an eligible zone established the incident."
    if reason == "insufficient_spatial_evidence":
        return "The notice has no localized detection that can be checked against the configured zones."
    if reason in {"ignored_zone", "depth_ignore_zone"}:
        return "Activity was only in an ignored zone, so it did not establish or prolong an incident."
    if reason == "ineligible_zone":
        return "Activity was not in a zone that can establish an incident."
    if reason == "outside_incident_zone":
        return "Activity was outside the zones that can establish an incident."
    if reason == "below_confidence":
        return "Activity was below the confidence required to establish an incident."
    if reason == "stationary_scene_context":
        return "A stationary object already known at this location did not establish an incident."
    return "Activity was not in a zone that can establish an incident."


def _assign_known_stationary_roles(
    samples,
    memory,
    *,
    event_key: str,
    observed_at_epoch: float | None,
) -> None:
    """A box with no role that sits on a known stationary subject is scene context.

    An explicit role is left alone. The current event does not count as a prior,
    and a snapshot recorded before this policy does not consult the live ledger.
    """
    if memory is None:
        return
    thresholds = stationary_object_policy("balanced")
    for sample in samples or []:
        if not isinstance(sample, dict):
            continue
        camera_id = str(sample.get("camera_id") or "")
        sample_epoch = _number(sample.get("captured_epoch"))
        epoch = float(sample_epoch if observed_at_epoch is None else observed_at_epoch)
        if sample_epoch is not None:
            epoch = max(epoch, sample_epoch)
        for observation in sample.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload", observation)
            if not isinstance(payload, dict) or str(payload.get("activity_role") or ""):
                continue
            label = str(payload.get("label") or "").strip().lower()
            box = normalized_box(payload)
            if not label or box is None:
                continue
            matched, priors, _age = memory.match(
                camera_id=camera_id,
                label=label,
                box=box,
                event_key=event_key,
                observed_at_epoch=epoch,
                min_iou=thresholds.scene_memory_min_iou,
                ttl_seconds=thresholds.scene_memory_ttl_seconds,
            )
            if matched and priors >= thresholds.scene_memory_min_prior_sightings:
                payload["activity_role"] = "scene_context"


def _notice_activity_assessment(samples, notice):
    """Replay the alert path's physical assessment, never its notification rules.

    Raw scene observations precede alert filtering. Their presence or stale
    incident_eligible flag cannot override a required motion assessment.
    Older notices retain their recorded policy when no snapshot is available.
    """
    policy = (notice or {}).get("activity_policy")
    if not isinstance(policy, dict) or not policy.get("require_motion_correlation"):
        return None
    objects = []
    noticed = _number(notice.get("epoch"))
    complete = [sample for sample in samples if isinstance(sample, dict) and sample.get("status") == "complete"]
    for identifier, observation, width, height in _iter_observations(complete):
        payload = dict(observation.get("payload", observation))
        captured = _number(payload.get("captured_at_epoch", payload.get("frame_captured_at_epoch")))
        if not identifier or captured is None or noticed is None or abs(captured - noticed) > NOTICE_VERIFICATION_SECONDS:
            continue
        payload.update(id=identifier, detection_frame_width=width, detection_frame_height=height)
        objects.append(payload)
    admitted, diagnostics = assess_motion_objects(
        objects, {"features": notice.get("features") or {}}, policy.get("alignment"),
    )
    return {
        "version": 1,
        "supporting_observation_ids": sorted({item["id"] for item in admitted}),
        "observations": [{"observation_id": item["id"],
                          "supported": item["motion_correlated"],
                          "reason": item["motion_correlation"] if item["motion_correlated"] else "object_not_motion_correlated"}
                         for item in objects],
        "diagnostics": diagnostics,
    }


def evaluate_scene_establishment(
    samples,
    *,
    policy: dict[str, Any] | None = None,
    notice: dict[str, Any] | None = None,
    scene_context_memory=None,
    event_key: str = "",
    observed_at_epoch: float | None = None,
) -> dict[str, Any]:
    """Combine physical evidence with the snapshotted zone policy.

    An unlocalized camera or motion notice is not spatial evidence. When ignore
    zones or required incident zones exist, that notice establishes activity
    only if retained evidence on the same camera is establishment-eligible.
    A snapshotted motion-correlation requirement is shared with alert assessment;
    a box's presence cannot satisfy it. Measured motion that names no observation
    cannot satisfy a spatial restriction.
    """
    if isinstance(policy, dict) and "stationary_subject_presence" in policy and policy.get("object_activity_attribution") != "off":
        _assign_known_stationary_roles(
            samples,
            scene_context_memory,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
        )
    activity = evaluate_scene_activity(
        samples,
        ignore_stationary_scene_context=stationary_presence_enforced(policy),
    )
    if not isinstance(policy, dict):
        policy = None
    restricting = policy_restricts(policy)
    notice_activity = _notice_activity_assessment(samples, notice)
    indexed = {}
    interpretations = []
    for identifier, observation, width, height in _iter_observations(samples):
        interpreted = interpret_establishment_observation(observation, policy, frame_width=width, frame_height=height)
        if identifier:
            indexed[identifier] = interpreted
        interpretations.append(interpreted)
    zone = {
        "policy_version": 1,
        "restricting": restricting,
        "require_incident_zone": bool(policy.get("require_incident_zone")) if isinstance(policy, dict) else False,
        "establishment_eligible": False,
        "reason": "full_frame" if not restricting else "insufficient_spatial_evidence",
        "observations": interpretations,
        "eligible_observation_ids": sorted(item["observation_id"] for item in interpretations if item["establishment_eligible"] and item["observation_id"]),
        "ignored_observation_ids": sorted(item["observation_id"] for item in interpretations if item["reason"] in {"ignored_zone", "depth_ignore_zone"} and item["observation_id"]),
    }
    result = {
        **{key: activity[key] for key in ("status", "reason", "summary", "activity_epoch", "supporting_observation_ids", "start_epoch") if key in activity},
        "diagnostics": activity.get("diagnostics") or {},
        "policy_version": activity.get("policy_version", 1) if policy is None else (
            ESTABLISHMENT_ZONE_POLICY_VERSION_V2
            if isinstance(policy, dict) and "stationary_subject_presence" in policy
            else ESTABLISHMENT_ZONE_POLICY_VERSION
        ),
        "physical_evidence": _compact_activity(activity),
        "zone_interpretation": zone,
        "witnesses": activity.get("witnesses") or [],
    }
    if any(item.get("stationary_counterfactual") for item in interpretations):
        result["diagnostics"] = {
            **(result.get("diagnostics") or {}),
            "stationary_scene_context_counterfactual": True,
        }
    if notice_activity is not None:
        result["notice_activity"] = notice_activity
        result["policy_version"] = "establishment_activity_v3"
    if not restricting and (notice_activity is None or activity["status"] == "supported"):
        if notice and result.get("status") != "supported":
            source = "camera" if notice.get("source") == "camera" else "motion"
            result.update(
                status="supported",
                reason="admitted_camera_notice",
                summary="The camera reported activity." if source == "camera" else "The motion pipeline reported activity.",
                activity_epoch=notice.get("epoch"),
                supporting_observation_ids=[],
                evidence_kind="camera_reported",
            )
            zone.update(establishment_eligible=True, reason="full_frame")
        elif result.get("status") == "supported":
            zone.update(establishment_eligible=True, reason="full_frame")
        return result

    eligible_witnesses = []
    witness_interpretations = []
    for witness in activity.get("witnesses") or []:
        identifiers = [str(item) for item in witness.get("observation_ids") or []]
        located = [indexed[item] for item in identifiers if item in indexed]
        witness_interpretations.extend(located)
        eligible = [item["observation_id"] for item in located if _admits_activity(item)]
        # Motion with no observation cannot prove it occurred outside an ignore zone.
        if eligible:
            eligible_witnesses.append((float(witness["epoch"]), eligible))
    if eligible_witnesses:
        eligible_ids = sorted({item for _, ids in eligible_witnesses for item in ids})
        zone.update(establishment_eligible=True, reason="incident_zone" if any(indexed[item]["reason"] == "incident_zone" for item in eligible_ids) else "full_frame")
        result.update(
            status="supported",
            reason="eligible_zone_activity",
            summary=_zone_summary(zone["reason"], established=True, notice=False),
            activity_epoch=max(epoch for epoch, _ in eligible_witnesses),
            supporting_observation_ids=eligible_ids,
            evidence_kind=(activity.get("diagnostics") or {}).get("evidence_kind") or "video_verified_activity",
        )
        return result
    if notice_activity is not None and not interpretations and activity["status"] in {"pending", "incomplete"}:
        return result
    if activity.get("status") == "supported" or notice:
        considered = witness_interpretations or interpretations
        reason = _rejection_reason(considered)
        # A notice must satisfy its recorded activity policy as well as the
        # zone policy. It cannot override an ineligible physical witness.
        if notice and activity.get("status") != "supported":
            eligible_ids = sorted(item["observation_id"] for item in interpretations if _admits_activity(item) and item["observation_id"])
            if notice_activity is not None:
                supporting = set(notice_activity["supporting_observation_ids"])
                verified_ids = [identifier for identifier in eligible_ids if identifier in supporting]
                if eligible_ids and not verified_ids:
                    result.update(
                        status=activity["status"] if activity["status"] in {"pending", "incomplete"} else "unsupported",
                        reason="object_not_motion_correlated",
                        summary="Detected objects did not explain the triggering motion; relevant activity remains unconfirmed.",
                        activity_epoch=None, supporting_observation_ids=[], evidence_kind="none",
                    )
                    zone.update(establishment_eligible=False, reason="object_not_motion_correlated")
                    return result
                eligible_ids = verified_ids
            if eligible_ids:
                epochs = []
                noticed = _number(notice.get("epoch"))
                verifying = set(eligible_ids)
                for identifier, observation, _, _ in _iter_observations(samples):
                    if identifier in verifying:
                        payload = observation.get("payload", observation)
                        captured = _number(payload.get("captured_at_epoch"))
                        if captured is not None and (noticed is None or captured <= noticed + NOTICE_VERIFICATION_SECONDS):
                            epochs.append(captured)
                zone_reason = "incident_zone" if any(indexed[key]["reason"] == "incident_zone" for key in eligible_ids) else "full_frame"
                zone.update(establishment_eligible=True, reason=zone_reason)
                result.update(
                    status="supported",
                    reason="verified_object_activity" if notice_activity is not None else "verified_camera_notice",
                    summary="Object activity explained the triggering motion in an eligible zone." if notice_activity is not None else _zone_summary("incident_zone", established=True, notice=True),
                    activity_epoch=max(epochs) if epochs else notice.get("epoch"),
                    supporting_observation_ids=eligible_ids,
                    evidence_kind="motion_correlated_object" if notice_activity is not None else "camera_reported",
                )
                return result
        zone.update(establishment_eligible=False, reason=reason)
        result.update(
            status="unsupported",
            reason=reason,
            summary=_zone_summary(reason, established=False, notice=bool(notice)),
            activity_epoch=None,
            supporting_observation_ids=[],
            evidence_kind="zone_restricted",
        )
    return result
