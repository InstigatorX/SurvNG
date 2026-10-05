"""Shared, replayable object-to-motion assessment for incidents and alerts.

Callers supply copies when the original detector observations must stay immutable.
Notification preferences are deliberately outside this assessment.
"""
from __future__ import annotations

import math
from typing import Any

from .object_motion import (
    MAXIMUM_MOVEMENT_RATIO,
    MINIMUM_MOVEMENT_RATIO,
    temporal_object_motion_evidence,
)

MOTION_REGION_MARGIN_RATIO = 0.035


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


def assess_motion_objects(
    objects: list[dict[str, Any]],
    qualification: dict[str, Any],
    alignment: dict[str, Any] | None = None,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
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
    for detected in objects:
        snapshot_visible = detected.get("snapshot_visible") is not False
        evidence = temporal_object_motion_evidence(
            detected,
            frame_width=frame_width if snapshot_visible else None,
            frame_height=frame_height if snapshot_visible else None,
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
            and evidence.movement_extent_ratio >= evidence.path_threshold
        )
        semantic_tier = str(detected.get("semantic_tier") or "standard")
        standard_semantic = semantic_tier == "standard"
        stable_geometry = bool(
            evidence.track_observations >= 3
            and evidence.displacement_ratio < evidence.movement_threshold
            and evidence.movement_extent_ratio < evidence.path_threshold
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
        # Excursion can establish out-and-back movement with zero net travel.
        # New evidence uses a bounded excursion, never accumulated box jitter;
        # legacy records retain their historical aggregate-path interpretation.
        spatial_path = bool(
            spatial
            and evidence.temporal_evidence_available
            and evidence.movement_extent_ratio >= evidence.path_threshold
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
