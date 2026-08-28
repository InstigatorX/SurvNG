from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import time
from typing import Any, Callable, Protocol

from ..detector import detection_failure
from ..domain_events import IncidentCreated, ObjectDetected
from ..face_candidates import FaceCandidate
from ..object_activity import AttributionMode, ObjectActivityAttributor
from ..object_motion import (
    MAXIMUM_MOVEMENT_RATIO,
    MINIMUM_MOVEMENT_RATIO,
    temporal_object_motion_evidence,
)
from .context import Frame


MotionDetectionProvider = Callable[[datetime], Any]
MotionEvidenceDetectionProvider = Callable[[datetime, dict[str, Any]], Any]
MotionSnapshotWriter = Callable[[Frame, datetime], str]
MotionEventCallback = Callable[[str, dict[str, Any]], None]
MotionObjectSerializer = Callable[[list[dict[str, Any]]], str]
FaceCandidateSink = Callable[[int, str, str, list[dict[str, Any]]], int]
RefinementCoverPromoter = Callable[..., dict[str, Any] | None]


LOGGER = logging.getLogger(__name__)
MOTION_REGION_MARGIN_RATIO = 0.035


def _route_origin(
    qualification: dict[str, Any],
) -> tuple[str, int] | None:
    features = qualification.get("features")
    if not isinstance(features, dict):
        return None
    watch = features.get("route_detection_watch")
    if not isinstance(watch, dict):
        return None
    camera_id = str(
        watch.get("origin_camera_id") or watch.get("source_camera_id") or ""
    ).strip()
    try:
        event_id = int(
            watch.get("origin_event_id") or watch.get("source_event_id") or 0
        )
    except (TypeError, ValueError):
        return None
    return (camera_id, event_id) if camera_id and event_id > 0 else None


def _intersects_motion_region(
    box: tuple[float, float, float, float],
    regions: list[object],
) -> bool:
    x1, y1, x2, y2 = box
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            continue
        try:
            rx1, ry1, rx2, ry2 = (float(value) for value in region)
        except (TypeError, ValueError):
            continue
        rx1 -= MOTION_REGION_MARGIN_RATIO
        ry1 -= MOTION_REGION_MARGIN_RATIO
        rx2 += MOTION_REGION_MARGIN_RATIO
        ry2 += MOTION_REGION_MARGIN_RATIO
        if min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1):
            return True
    return False


def _aligned_motion_regions(
    regions: list[object],
    alignment: dict[str, Any],
) -> list[list[float]]:
    scale_x = float(alignment.get("scale_x", 1.0))
    scale_y = float(alignment.get("scale_y", 1.0))
    offset_x = float(alignment.get("offset_x", 0.0))
    offset_y = float(alignment.get("offset_y", 0.0))
    aligned: list[list[float]] = []
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in region)
        except (TypeError, ValueError):
            continue
        aligned.append([
            max(0.0, min(1.0, x1 * scale_x + offset_x)),
            max(0.0, min(1.0, y1 * scale_y + offset_y)),
            max(0.0, min(1.0, x2 * scale_x + offset_x)),
            max(0.0, min(1.0, y2 * scale_y + offset_y)),
        ])
    return aligned


def motion_correlated_objects(
    frame: Frame,
    objects: list[dict[str, Any]],
    qualification: dict[str, Any],
    alignment: dict[str, Any] | None = None,
    *,
    depth_attribution_mode: str = "off",
    depth_shadow_maximum_m: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep objects that spatially or temporally explain an EMA trigger.

    ``depth_attribution_mode='shadow'`` emits decision-scoped diagnostics for
    enriched refinement objects.  It deliberately never changes admission;
    live qualification has no depth producer on this path.
    """
    features = qualification.get("features")
    regions = features.get("motion_regions", []) if isinstance(features, dict) else []
    if not isinstance(regions, list):
        regions = []
    alignment = dict(alignment or {})
    alignment_reliable = bool(alignment.get("reliable", True))
    if alignment_reliable:
        regions = _aligned_motion_regions(regions, alignment)
    correlated: list[dict[str, Any]] = []
    spatial_matches = 0
    temporal_matches = 0
    temporal_path_matches = 0
    temporal_path_only_matches = 0
    new_appearance_matches = 0
    stationary_spatial_rejections = 0
    height, width = frame.shape[:2]
    for detected in objects:
        evidence = temporal_object_motion_evidence(
            detected,
            frame_width=width,
            frame_height=height,
        )
        box = evidence.normalized_box
        spatial = bool(
            alignment_reliable
            and box is not None
            and _intersects_motion_region(box, regions)
        )
        temporal = evidence.displacement_ratio >= evidence.movement_threshold
        temporal_path = bool(
            evidence.temporal_evidence_available
            and evidence.path_ratio >= evidence.path_threshold
        )
        semantic_tier = str(detected.get("semantic_tier") or "standard")
        standard_semantic = semantic_tier == "standard"
        stable_geometry = bool(
            evidence.track_observations >= 3
            and evidence.displacement_ratio < evidence.movement_threshold
            and evidence.path_ratio < evidence.path_threshold
        )
        spatial_fallback = bool(
            standard_semantic and spatial and not evidence.temporal_evidence_available
        )
        appearance_match = bool(
            evidence.robust_new_appearance
            and alignment_reliable
            and spatial
            and not stable_geometry
        )
        alignment_fallback = bool(
            standard_semantic
            and not alignment_reliable
            and not evidence.temporal_evidence_available
        )
        # A short recorded sequence can begin and end at nearly the same point
        # while a real object walks through the EMA region.  Permit that only
        # when spatial evidence also agrees and the travelled path is well
        # beyond ordinary detector-box jitter.  A stationary object therefore
        # still cannot explain unrelated motion beside or behind it.
        spatial_path = bool(
            spatial
            and evidence.temporal_evidence_available
            and evidence.path_ratio >= evidence.path_threshold
        )
        motion_correlated = bool(
            temporal
            or evidence.zone_entry
            or temporal_path
            or spatial_path
            or spatial_fallback
            or appearance_match
            or alignment_fallback
        )
        if (
            depth_attribution_mode == "shadow"
            and isinstance(detected.get("depth_stats"), dict)
        ):
            depth_stats = detected["depth_stats"]
            try:
                median_m = float(depth_stats.get("median_m"))
            except (TypeError, ValueError):
                median_m = None
            valid_depth = median_m is not None and math.isfinite(median_m) and median_m > 0
            near_depth = bool(valid_depth and median_m <= depth_shadow_maximum_m)
            would_admit = bool(
                near_depth
                and alignment_reliable
                and spatial
                and not stable_geometry
                and not motion_correlated
            )
            provenance = {
                key: detected[key]
                for key in (
                    "frame_captured_at_epoch",
                    "captured_at",
                    "frame_offset_s",
                    "temporal_sample_count",
                    "temporal_incident_observations",
                )
                if detected.get(key) is not None
            }
            detected["depth_attribution"] = {
                "mode": "shadow",
                "decision_scoped": True,
                "median_m": median_m,
                "valid_depth": valid_depth,
                "near_depth": near_depth,
                "maximum_m": depth_shadow_maximum_m,
                "alignment_reliable": alignment_reliable,
                "spatial_match": spatial,
                "stable_geometry": stable_geometry,
                "normal_motion_correlated": motion_correlated,
                "would_admit": would_admit,
                "provenance": provenance,
            }
        detected["motion_correlated"] = motion_correlated
        detected["motion_correlation"] = (
            "temporal" if temporal else
            "zone_entry" if evidence.zone_entry else
            "spatial_path" if spatial_path else
            "temporal_path" if temporal_path else
            "appearance" if appearance_match else
            "alignment_unverified" if alignment_fallback else
            "spatial" if spatial_fallback else
            "none"
        )
        detected["motion_correlation_threshold"] = round(
            evidence.movement_threshold,
            5,
        )
        detected["motion_correlation_eligible"] = motion_correlated
        detected["motion_temporal_evidence_available"] = (
            evidence.temporal_evidence_available
        )
        if motion_correlated:
            correlated.append(detected)
            spatial_matches += int(spatial)
            temporal_matches += int(temporal)
            temporal_path_matches += int(spatial_path)
            temporal_path_only_matches += int(temporal_path and not spatial)
            new_appearance_matches += int(appearance_match)
        else:
            # Preserve the detection as diagnostic evidence without allowing
            # an unrelated stationary object to become an incident label.
            detected["incident_eligible"] = False
            existing_reasons = detected.get("incident_ineligible_reasons")
            reasons = (
                [str(value) for value in existing_reasons]
                if isinstance(existing_reasons, list)
                else [str(existing_reasons)] if existing_reasons else []
            )
            detected["incident_ineligible_reasons"] = list(dict.fromkeys([
                *reasons,
                "object_not_motion_correlated",
            ]))
            stationary_spatial_rejections += int(
                spatial and evidence.temporal_evidence_available
            )
    return correlated, {
        "required": True,
        "motion_region_count": len(regions),
        "eligible_object_count": len(objects),
        "correlated_object_count": len(correlated),
        "spatial_match_count": spatial_matches,
        "temporal_match_count": temporal_matches,
        "temporal_path_match_count": temporal_path_matches,
        "temporal_path_only_match_count": temporal_path_only_matches,
        "new_appearance_match_count": new_appearance_matches,
        "stationary_spatial_rejection_count": stationary_spatial_rejections,
        "minimum_temporal_movement_ratio": MAXIMUM_MOVEMENT_RATIO,
        "adaptive_minimum_temporal_movement_ratio": MINIMUM_MOVEMENT_RATIO,
        "region_margin_ratio": MOTION_REGION_MARGIN_RATIO,
        "alignment_reliable": alignment_reliable,
        "alignment_mode": str(alignment.get("mode") or "legacy_identity"),
        "alignment_confidence": float(alignment.get("confidence", 1.0)),
    }


def _admit_semantic_rescues(
    frame: Frame,
    objects: list[dict[str, Any]],
    qualification: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """Reject below-threshold candidates after recording causal context.

    The configured object confidence is the incident admission floor. Causal
    motion may explain a candidate, but must not override that operator policy.
    """
    candidates = [
        item for item in objects
        if item.get("label") and item.get("semantic_tier") == "rescue_candidate"
    ]
    if not candidates:
        return {
            "policy": "causal_v1",
            "candidate_count": 0,
            "admitted_count": 0,
            "rejected_count": 0,
        }
    correlated, correlation = motion_correlated_objects(
        frame,
        candidates,
        qualification,
        alignment,
        depth_attribution_mode="shadow",
    )
    admitted_ids = {id(item) for item in correlated}
    admitted = 0
    for detected in candidates:
        activity_role = str(detected.get("activity_role") or "indeterminate")
        zone_allowed = detected.get("spatial_zone_eligible") is True
        confidence_allowed = bool(detected.get("confidence_eligible") is True)
        qualifying_observations = max(
            0,
            int(detected.get("temporal_incident_observations") or 0),
        )
        required_observations = max(
            1,
            int(detected.get("temporal_required_observations") or 1),
        )
        confirmations_allowed = qualifying_observations >= required_observations
        eligible = bool(
            confidence_allowed
            and confirmations_allowed
            and id(detected) in admitted_ids
            and zone_allowed
            and activity_role != "scene_context"
        )
        detected["semantic_rescue_admitted"] = eligible
        detected["semantic_rescue_policy"] = "causal_v1"
        detected["semantic_previous_policy_admitted"] = True
        detected["semantic_final_admission"] = (
            "rescue" if eligible else "rejected"
        )
        detected["incident_eligible"] = eligible
        detected["temporal_eligible"] = eligible
        reasons = detected.get("incident_ineligible_reasons")
        normalized = (
            [str(value) for value in reasons]
            if isinstance(reasons, list)
            else [str(reasons)] if reasons else []
        )
        normalized = [reason for reason in normalized if reason != "pending_causal_confirmation"]
        if eligible:
            admitted += 1
            detected["incident_ineligible_reasons"] = normalized
            detected["incident_admission_reason"] = "temporal_rescue_with_causal_motion"
        else:
            rejection = (
                "below_confidence_threshold"
                if not confidence_allowed
                else "insufficient_qualifying_confirmations"
                if not confirmations_allowed
                else "stationary_scene_context"
                if activity_role == "scene_context"
                else "outside_incident_zone"
                if not zone_allowed
                else "low_confidence_without_causal_motion"
            )
            detected["incident_ineligible_reasons"] = list(dict.fromkeys([
                *normalized,
                rejection,
            ]))
            detected["incident_admission_reason"] = rejection
    return {
        "policy": "causal_v1",
        "candidate_count": len(candidates),
        "admitted_count": admitted,
        "rejected_count": len(candidates) - admitted,
        "correlation": correlation,
    }


def _depth_attribution_summary(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return bounded, decision-scoped depth diagnostics for motion audits."""
    entries: list[dict[str, Any]] = []
    for detected in objects:
        attribution = detected.get("depth_attribution")
        if not isinstance(attribution, dict):
            continue
        entries.append({
            "label": str(detected.get("label") or ""),
            "box": dict(detected["box"]) if isinstance(detected.get("box"), dict) else None,
            "median_m": attribution.get("median_m"),
            "valid_depth": bool(attribution.get("valid_depth")),
            "near_depth": bool(attribution.get("near_depth")),
            "would_admit": bool(attribution.get("would_admit")),
            "alignment_reliable": bool(attribution.get("alignment_reliable")),
            "spatial_match": bool(attribution.get("spatial_match")),
            "stable_geometry": bool(attribution.get("stable_geometry")),
            "provenance": dict(attribution.get("provenance") or {}),
        })
    if not entries:
        return None
    return {
        "schema_version": 1,
        "evaluated_count": len(entries),
        "valid_depth_count": sum(item["valid_depth"] for item in entries),
        "near_depth_count": sum(item["near_depth"] for item in entries),
        "would_admit_count": sum(item["would_admit"] for item in entries),
        "alignment_reliable_count": sum(item["alignment_reliable"] for item in entries),
        "spatial_match_count": sum(item["spatial_match"] for item in entries),
        "stable_geometry_count": sum(item["stable_geometry"] for item in entries),
        "objects": entries[:12],
        "truncated": len(entries) > 12,
    }


class MotionEventStore(Protocol):
    def add_event(
        self,
        camera_id: str,
        kind: str,
        topic: str = "",
        message: str = "",
        snapshot_path: str = "",
        recording_path: str = "",
        objects_json: str = "[]",
        created_at: str | None = None,
        detection_intent_id: str | None = None,
        route_origin_camera_id: str | None = None,
        route_origin_event_id: int | None = None,
    ) -> dict[str, Any]:
        ...

    def add_motion_audit(
        self,
        camera_id: str,
        snapshot_path: str,
        created_at: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        ...

    def refine_event_evidence(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        objects_json: str,
    ) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True, slots=True)
class MotionDecisionOutcome:
    event_id: int | None
    snapshot_path: str
    object_detected: bool | None
    detected_objects: tuple[dict[str, Any], ...] = ()
    rejection_reason: str = ""
    motion_correlation: dict[str, Any] | None = None
    refinement_pending: bool = False
    processing_timing: dict[str, Any] | None = None
    object_activity: dict[str, Any] | None = None
    depth_attribution: dict[str, Any] | None = None
    cover_promoted: bool = False
    cover_promotion_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "snapshot_path": self.snapshot_path,
            "object_detected": self.object_detected,
            "rejection_reason": self.rejection_reason,
            "motion_correlation": self.motion_correlation,
            "refinement_pending": self.refinement_pending,
            "processing_timing": self.processing_timing,
            "object_activity": self.object_activity,
            "depth_attribution": self.depth_attribution,
            "cover_promoted": self.cover_promoted,
            "cover_promotion_reason": self.cover_promotion_reason,
        }


class MotionDecisionHandler:
    """Consumes a motion decision and owns downstream event side effects."""

    def __init__(
        self,
        camera_id: str,
        events: MotionEventStore,
        detection_provider: MotionDetectionProvider,
        snapshot_writer: MotionSnapshotWriter,
        object_serializer: MotionObjectSerializer,
        initial_detection_provider: MotionDetectionProvider | None = None,
        initial_evidence_detection_provider: MotionEvidenceDetectionProvider | None = None,
        event_callback: MotionEventCallback | None = None,
        activity_attributor: ObjectActivityAttributor | None = None,
        face_candidate_sink: FaceCandidateSink | None = None,
        spatial_alignment: dict[str, Any] | None = None,
        refinement_cover_promoter: RefinementCoverPromoter | None = None,
        route_admission_callback: Callable[[str, str, int], object] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.detection_provider = detection_provider
        self.initial_detection_provider = initial_detection_provider
        self.initial_evidence_detection_provider = initial_evidence_detection_provider
        self.snapshot_writer = snapshot_writer
        self.object_serializer = object_serializer
        self.event_callback = event_callback
        self.activity_attributor = activity_attributor
        self.face_candidate_sink = face_candidate_sink
        self.spatial_alignment = dict(spatial_alignment or {"reliable": True})
        self.refinement_cover_promoter = refinement_cover_promoter
        self.route_admission_callback = route_admission_callback

    def activity_status(self) -> dict[str, Any]:
        if self.activity_attributor is None:
            return {"mode": "off", "evaluated": 0}
        return self.activity_attributor.status()

    def reconfigure_activity_attribution(self, mode: AttributionMode) -> None:
        if self.activity_attributor is not None:
            self.activity_attributor.reconfigure(mode)

    def reconfigure_stationary_tolerance(self, tolerance: str) -> None:
        if self.activity_attributor is not None:
            self.activity_attributor.reconfigure_stationary_tolerance(tolerance)

    def handle(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        return self._handle_with_provider(
            self.initial_detection_provider or self.detection_provider,
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
            evidence_provider=self.initial_evidence_detection_provider,
        )

    def refine(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        existing_event_id: int | None,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        """Run delayed evidence and atomically enrich or create the incident."""
        return self._handle_with_provider(
            self.detection_provider,
            topic,
            message,
            event_at,
            qualification,
            existing_event_id=existing_event_id,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )

    def _handle_with_provider(
        self,
        provider: MotionDetectionProvider,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        existing_event_id: int | None = None,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
        evidence_provider: MotionEvidenceDetectionProvider | None = None,
    ) -> MotionDecisionOutcome:
        detection_started = time.monotonic()
        detection_started_epoch = time.time()
        provider_result = (
            evidence_provider(event_at, qualification)
            if evidence_provider is not None
            else provider(event_at)
        )
        frame, objects, recording_path = provider_result
        frame_captured_at_epoch = getattr(
            provider_result,
            "frame_captured_at_epoch",
            None,
        )
        if not isinstance(frame_captured_at_epoch, (int, float)) or not math.isfinite(
            float(frame_captured_at_epoch)
        ):
            frame_captured_at_epoch = None
        frame_source = str(getattr(provider_result, "frame_source", "") or "")
        frame_timestamp_exact = bool(
            getattr(provider_result, "frame_timestamp_exact", False)
        )
        if frame_captured_at_epoch is not None:
            for detected in objects:
                if not isinstance(detected, dict) or not detected.get("label"):
                    continue
                detected.setdefault(
                    "frame_captured_at_epoch",
                    round(float(frame_captured_at_epoch), 6),
                )
                if frame_source:
                    detected.setdefault("frame_source", frame_source)
                detected.setdefault("frame_timestamp_exact", frame_timestamp_exact)
        processed_at = datetime.now(timezone.utc)
        normalized_event_at = (
            event_at.replace(tzinfo=timezone.utc)
            if event_at.tzinfo is None
            else event_at.astimezone(timezone.utc)
        )
        workflow_ms = round(
            (time.monotonic() - detection_started) * 1000,
            3,
        )
        qualification["object_detection_workflow_ms"] = workflow_ms
        timings = getattr(provider_result, "timings_ms", None)
        if isinstance(timings, dict):
            qualification["object_detection_phases_ms"] = {
                str(key): round(max(0.0, float(value)), 3)
                for key, value in timings.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        received_at_epoch = qualification.get("trigger_received_at_epoch")
        if isinstance(received_at_epoch, (int, float)):
            qualification["decision_queue_wait_ms"] = round(
                max(0.0, (detection_started_epoch - float(received_at_epoch)) * 1000.0),
                3,
            )
        qualification["processed_at"] = processed_at.isoformat()
        qualification["event_processing_delay_seconds"] = round(
            max(0.0, (processed_at - normalized_event_at).total_seconds()),
            3,
        )
        if frame is None:
            objects = [{"status": "no_recorded_frame"}]
        processing_timing = {
            "workflow_ms": workflow_ms,
            "decision_queue_wait_ms": qualification.get("decision_queue_wait_ms"),
            "phases_ms": qualification.get("object_detection_phases_ms", {}),
        }

        detection_completed = frame is not None and not detection_failure(objects)

        if existing_event_id is not None and not detection_completed:
            # Refinement is additive. A transient decoder/detector failure must
            # never replace a known-good provisional snapshot or its objects
            # with empty/status-only evidence.
            qualification["refinement_evidence_preserved"] = True
            return MotionDecisionOutcome(
                event_id=int(existing_event_id),
                snapshot_path="",
                object_detected=None,
                detected_objects=(),
                rejection_reason="refinement_unavailable_preserved",
                refinement_pending=False,
                processing_timing=processing_timing,
            )

        activity_summary: dict[str, Any] | None = None
        if (
            self.activity_attributor is not None
            and self.activity_attributor.mode != "off"
        ):
            admissions = self.activity_attributor.admit(
                objects,
                qualification,
                event_key=normalized_event_at.isoformat(),
                observed_at_epoch=normalized_event_at.timestamp(),
            )
            objects = [admission.stored_observation() for admission in admissions]
            activity_objects = [
                {
                    "label": admission.observation.get("label"),
                    "role": admission.attribution.role.value,
                    "confidence": round(admission.attribution.confidence, 4),
                    "admitted": admission.admitted,
                    "reason": admission.reason,
                }
                for admission in admissions
                if admission.observation.get("label")
            ]
            activity_summary = {
                "mode": self.activity_attributor.mode,
                "observed": len(activity_objects),
                "detector_admitted": sum(
                    admission.detector_eligible for admission in admissions
                    if admission.observation.get("label")
                ),
                "zone_rejected": sum(
                    admission.observation.get("confidence_eligible") is not False
                    and admission.observation.get("zone_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "confidence_rejected": sum(
                    admission.observation.get("confidence_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "temporal_rejected": sum(
                    admission.observation.get("zone_eligible") is not False
                    and admission.observation.get("temporal_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "active": sum(item["role"] == "active" for item in activity_objects),
                "scene_context": sum(
                    item["role"] == "scene_context" for item in activity_objects
                ),
                "indeterminate": sum(
                    item["role"] == "indeterminate" for item in activity_objects
                ),
                "admitted": sum(bool(item["admitted"]) for item in activity_objects),
                "scene_context_suppressed": sum(
                    item["reason"] == "stationary_scene_context"
                    for item in activity_objects
                ),
                "objects": activity_objects,
            }
            qualification["object_activity_attribution"] = activity_summary

        if frame is not None:
            rescue_summary = _admit_semantic_rescues(
                frame,
                objects,
                qualification,
                self.spatial_alignment,
            )
            rescue_summary["trigger_source"] = str(
                qualification.get("trigger_source") or "unknown"
            )
            qualification["semantic_rescue"] = rescue_summary
            if activity_summary is not None:
                activity_summary["semantic_rescue"] = rescue_summary
                activity_summary["final_admitted"] = sum(
                    bool(item.get("label") and item.get("incident_eligible") is not False)
                    for item in objects
                )
            if self.activity_attributor is not None:
                self.activity_attributor.record_semantic_rescue(
                    source=rescue_summary["trigger_source"],
                    candidates=int(rescue_summary["candidate_count"]),
                    admitted=int(rescue_summary["admitted_count"]),
                    rejected=int(rescue_summary["rejected_count"]),
                )

        eligible_objects = [
            detected
            for detected in objects
            if detected.get("label") and detected.get("incident_eligible") is not False
        ]
        correlation: dict[str, Any] | None = None
        uncorrelated_eligible_objects = 0
        if require_motion_correlation and frame is not None:
            uncorrelated_eligible_objects = len(eligible_objects)
            eligible_objects, correlation = motion_correlated_objects(
                frame,
                eligible_objects,
                qualification,
                self.spatial_alignment,
                depth_attribution_mode="shadow",
            )
            uncorrelated_eligible_objects -= len(eligible_objects)
            qualification["motion_correlation"] = correlation
        depth_attribution = _depth_attribution_summary(objects)
        verification_candidate = bool(qualification.get("suppression_verification_candidate"))
        if qualification.get("borderline_candidate"):
            qualification["rescued_by_object"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)
        if verification_candidate:
            qualification["suppression_verification_rescued"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)

        snapshot_path = ""
        if frame is not None:
            snapshot_at = (
                datetime.fromtimestamp(float(frame_captured_at_epoch), timezone.utc)
                if frame_captured_at_epoch is not None
                else event_at
            )
            snapshot_path = self.snapshot_writer(frame, snapshot_at)

        if require_eligible_object and not eligible_objects:
            rejection_reason = (
                "object_not_motion_correlated"
                if require_motion_correlation and uncorrelated_eligible_objects > 0
                else "no_eligible_object"
            )
            cover_promoted = False
            cover_promotion_reason = ""
            if (
                rejection_reason == "object_not_motion_correlated"
                and existing_event_id is not None
                and snapshot_path
                and frame is not None
                and frame_source == "recorded_main"
                and self.refinement_cover_promoter is not None
            ):
                frame_height, frame_width = frame.shape[:2]
                try:
                    promoted = self.refinement_cover_promoter(
                        int(existing_event_id),
                        snapshot_path=snapshot_path,
                        recording_path=recording_path,
                        captured_at=(
                            float(frame_captured_at_epoch)
                            if frame_captured_at_epoch is not None
                            else normalized_event_at.timestamp()
                        ),
                        frame_width=int(frame_width),
                        frame_height=int(frame_height),
                        cover_objects=[
                            dict(item)
                            for item in objects
                            if isinstance(item, dict) and item.get("label")
                        ],
                        source=frame_source,
                        timestamp_exact=frame_timestamp_exact,
                    )
                    cover_promoted = promoted is not None
                    cover_promotion_reason = (
                        "compatible_recorded_refinement"
                        if cover_promoted
                        else "refinement_cover_not_eligible"
                    )
                except Exception:
                    # Presentation enrichment is never allowed to make a
                    # completed security decision retry or fail terminally.
                    cover_promotion_reason = "refinement_cover_promotion_failed"
                    LOGGER.exception(
                        "refinement cover promotion failed for %s event %d",
                        self.camera_id,
                        int(existing_event_id),
                    )
                if cover_promoted:
                    self._publish("incident_update", {
                        "event_id": int(existing_event_id),
                        "camera_id": self.camera_id,
                        "timestamp": normalized_event_at.isoformat(),
                        "updated": True,
                        "reason": "cover_promoted",
                    })
            return MotionDecisionOutcome(
                event_id=existing_event_id,
                snapshot_path=snapshot_path,
                object_detected=False if detection_completed else None,
                detected_objects=tuple(eligible_objects),
                rejection_reason=rejection_reason,
                motion_correlation=correlation,
                refinement_pending=bool(
                    getattr(provider_result, "refinement_pending", False)
                ),
                processing_timing=processing_timing,
                object_activity=activity_summary,
                depth_attribution=depth_attribution,
                cover_promoted=cover_promoted,
                cover_promotion_reason=cover_promotion_reason,
            )

        stored_objects = [
            *objects,
            {"status": "motion_qualification", "motion_qualification": qualification},
        ]
        if bool(getattr(provider_result, "refinement_pending", False)):
            stored_objects.append({"status": "face_evidence_pending"})
        route_origin = _route_origin(qualification)
        route_admission_duplicate = False
        if existing_event_id is None:
            event = self.events.add_event(
                camera_id=self.camera_id,
                kind="motion",
                topic=topic,
                message=message,
                snapshot_path=snapshot_path,
                recording_path=recording_path,
                objects_json=self.object_serializer(stored_objects),
                created_at=event_at.isoformat(),
                detection_intent_id=(
                    str(qualification.get("detection_intent_id") or "") or None
                ),
                route_origin_camera_id=(
                    route_origin[0] if route_origin is not None else None
                ),
                route_origin_event_id=(
                    route_origin[1] if route_origin is not None else None
                ),
            )
            event_id = int(event["id"])
            event_created = bool(event.get("created", True))
            route_admission_duplicate = bool(
                route_origin is not None and not event_created
            )
        else:
            event_id = int(existing_event_id)
            event_created = False
            self.events.refine_event_evidence(
                event_id,
                snapshot_path=snapshot_path,
                recording_path=recording_path,
                objects_json=self.object_serializer(stored_objects),
            )
        if route_origin is not None and self.route_admission_callback is not None:
            try:
                self.route_admission_callback(
                    self.camera_id,
                    route_origin[0],
                    route_origin[1],
                )
            except Exception:
                # The durable event-store admission is authoritative. Runtime
                # watch cleanup can be recovered after restart and must never
                # turn a completed security decision into a retry.
                LOGGER.exception(
                    "route target cleanup failed for %s origin=%s/%d",
                    self.camera_id,
                    route_origin[0],
                    route_origin[1],
                )
        if route_admission_duplicate:
            return MotionDecisionOutcome(
                # The canonical event belongs to the winning decision. Do not
                # make this alternate job link its episode/audit/tracker to it.
                event_id=None,
                snapshot_path="",
                object_detected=False,
                detected_objects=(),
                rejection_reason="route_target_already_admitted",
                motion_correlation=correlation,
                refinement_pending=False,
                processing_timing=processing_timing,
                object_activity=activity_summary,
                depth_attribution=depth_attribution,
            )
        self._persist_face_candidates(
            event_id,
            event_at,
            tuple(getattr(provider_result, "face_candidates", ()) or ()),
        )
        if self.event_callback and event_created:
            self._publish(
                "incident",
                IncidentCreated(
                    event_id=event_id,
                    camera_id=self.camera_id,
                    timestamp=event_at.isoformat(),
                ).to_payload(),
            )

        detected_objects = [detected for detected in stored_objects if detected.get("label")]
        if self.event_callback and detected_objects and (
            existing_event_id is not None or event_created
        ):
            self._publish(
                "object",
                ObjectDetected(
                    event_id=event_id,
                    camera_id=self.camera_id,
                    timestamp=event_at.isoformat(),
                    snapshot_path=snapshot_path,
                    recording_path=recording_path,
                    objects=tuple(detected_objects),
                    incident_objects=tuple(eligible_objects),
                ).to_payload(),
            )
        return MotionDecisionOutcome(
            event_id=event_id,
            snapshot_path=snapshot_path,
            object_detected=bool(eligible_objects) if detection_completed else None,
            detected_objects=tuple(eligible_objects),
            motion_correlation=correlation,
            refinement_pending=bool(
                getattr(provider_result, "refinement_pending", False)
            ),
            processing_timing=processing_timing,
            object_activity=activity_summary,
            depth_attribution=depth_attribution,
        )

    def _persist_face_candidates(
        self,
        event_id: int,
        event_at: datetime,
        candidates: tuple[FaceCandidate, ...],
    ) -> None:
        if self.face_candidate_sink is None or not candidates:
            return
        persisted: list[dict[str, Any]] = []
        for candidate in candidates:
            box = candidate.box
            candidate_at = event_at + timedelta(seconds=candidate.offset_seconds)
            path = self.snapshot_writer(candidate.frame, candidate_at)
            if not path:
                continue
            persisted.append({
                "snapshot_path": path,
                "box": {
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                },
                "confidence": candidate.confidence,
                "track_id": candidate.track_id,
                "rank": candidate.rank,
                "offset_seconds": candidate.offset_seconds,
                "quality_score": candidate.quality_score,
                "quality": {
                    "sharpness": candidate.sharpness_score,
                    "exposure": candidate.exposure_score,
                    "edge_clearance": candidate.edge_clearance_ratio,
                },
                "detection_source": candidate.detection_source,
            })
        if persisted:
            try:
                self.face_candidate_sink(
                    event_id,
                    self.camera_id,
                    event_at.isoformat(),
                    persisted,
                )
            except Exception:
                LOGGER.exception(
                    "face candidate persistence failed for camera %s event %s",
                    self.camera_id,
                    event_id,
                )

    def record_audit(
        self,
        *,
        event_at: datetime,
        snapshot_path: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        audit = self.events.add_motion_audit(
            event_id=event_id,
            related_event_id=related_event_id,
            decision_id=decision_id,
            camera_id=self.camera_id,
            snapshot_path=snapshot_path,
            created_at=event_at.isoformat(),
            mode=mode,
            sensitivity=sensitivity,
            score=score,
            threshold=threshold,
            reason=reason,
            object_detected=object_detected,
            trigger_count=trigger_count,
            features=features,
            category=category,
        )
        if self.event_callback:
            self._publish("motion_audit", audit)
        return audit

    def _publish(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event_type, payload)
        except Exception:
            LOGGER.exception(
                "post-persistence %s notification failed for camera %s",
                event_type,
                self.camera_id,
            )


class MotionDecisionHandlerFactory:
    def __init__(
        self,
        events: MotionEventStore,
        object_serializer: MotionObjectSerializer,
        face_candidate_sink: FaceCandidateSink | None = None,
    ) -> None:
        self.events = events
        self.object_serializer = object_serializer
        self.face_candidate_sink = face_candidate_sink

    def create(
        self,
        camera_id: str,
        detection_provider: MotionDetectionProvider,
        snapshot_writer: MotionSnapshotWriter,
        initial_detection_provider: MotionDetectionProvider | None = None,
        initial_evidence_detection_provider: MotionEvidenceDetectionProvider | None = None,
        event_callback: MotionEventCallback | None = None,
        activity_attributor: ObjectActivityAttributor | None = None,
        spatial_alignment: dict[str, Any] | None = None,
        route_admission_callback: Callable[[str, str, int], object] | None = None,
    ) -> MotionDecisionHandler:
        return MotionDecisionHandler(
            camera_id=camera_id,
            events=self.events,
            detection_provider=detection_provider,
            initial_detection_provider=initial_detection_provider,
            initial_evidence_detection_provider=initial_evidence_detection_provider,
            snapshot_writer=snapshot_writer,
            object_serializer=self.object_serializer,
            event_callback=event_callback,
            activity_attributor=activity_attributor,
            face_candidate_sink=self.face_candidate_sink,
            spatial_alignment=spatial_alignment,
            route_admission_callback=route_admission_callback,
            refinement_cover_promoter=getattr(
                self.events,
                "promote_refinement_cover",
                None,
            ),
        )
