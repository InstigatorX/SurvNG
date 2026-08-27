const DEFAULT_VEHICLE_REID_LABELS = ["car", "truck", "bus", "motorcycle"];

/** True only for an explicit object index. `Number(null) === 0`, so null must not count. */
export function isValidObjectIndex(value) {
  if (value == null || value === "") return false;
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0;
}

export function visualSearchObjects(event) {
  return (Array.isArray(event?.objects) ? event.objects : [])
    .filter((object) => object && typeof object === "object"
      && String(object.label || "").trim()
      && object.snapshot_visible !== false);
}

export function visualSearchRequest({
  eventId,
  objectIndex,
  cameraFilter = "all",
  objectFilter = "all",
  startAt = "",
  endAt = "",
  limit = 50,
  sourceKinds = [],
  excludeAnchor = true,
}) {
  return {
    event_id: Number(eventId),
    object_index: Math.max(0, Number(objectIndex) || 0),
    camera_ids: cameraFilter && cameraFilter !== "all" ? [cameraFilter] : [],
    object_labels: objectFilter && objectFilter !== "all" ? [objectFilter] : [],
    start_at: String(startAt || ""),
    end_at: String(endAt || ""),
    limit: Math.max(1, Math.min(100, Number(limit) || 50)),
    source_kinds: Array.isArray(sourceKinds) ? sourceKinds.slice(0, 10) : [],
    exclude_anchor: Boolean(excludeAnchor),
  };
}

export function visualMatchLabel(matchStrength) {
  return ({
    strong_match: "Strong match",
    possible_match: "Possible match",
  })[String(matchStrength || "")] || "Visually similar";
}

export function appearanceCapableLabel(label, vehicleLabels = DEFAULT_VEHICLE_REID_LABELS) {
  const normalized = String(label || "").trim().toLowerCase();
  if (!normalized) return false;
  if (normalized === "person") return true;
  const vehicles = Array.isArray(vehicleLabels) && vehicleLabels.length
    ? vehicleLabels
    : DEFAULT_VEHICLE_REID_LABELS;
  return vehicles.map((item) => String(item || "").trim().toLowerCase()).includes(normalized);
}

export function resolveObjectTrackId(object, event = null) {
  const direct = Number(object?.track_id);
  if (Number.isInteger(direct) && direct > 0) return direct;
  const tracks = Array.isArray(event?.object_tracking?.tracks)
    ? event.object_tracking.tracks
    : [];
  const label = String(object?.label || "").trim().toLowerCase();
  if (!label || !tracks.length) return null;
  const matching = tracks.filter((track) => (
    String(track?.label || "").trim().toLowerCase() === label
    && Number.isInteger(Number(track?.track_id))
    && Number(track.track_id) > 0
  ));
  if (matching.length === 1) return Number(matching[0].track_id);
  const cover = Number(event?.object_tracking?.cover_primary_track_id);
  if (Number.isInteger(cover) && cover > 0 && matching.some((track) => Number(track.track_id) === cover)) {
    return cover;
  }
  return null;
}

export function appearanceMatchesPath(eventId, { hours = 24, limit = 12, trackId = null, crossCameraOnly = true } = {}) {
  const params = new URLSearchParams({
    hours: String(Math.max(0.25, Math.min(720, Number(hours) || 24))),
    limit: String(Math.max(1, Math.min(100, Number(limit) || 12))),
    cross_camera_only: crossCameraOnly ? "true" : "false",
  });
  if (Number.isInteger(Number(trackId)) && Number(trackId) > 0) {
    params.set("track_id", String(Number(trackId)));
  }
  return `/api/events/${Number(eventId)}/appearance-matches?${params.toString()}`;
}

export function normalizeAppearanceFindSimilarHits(matches, { requireVisuallySimilar = true } = {}) {
  return (Array.isArray(matches) ? matches : [])
    .filter((match) => {
      const eventId = Number(match?.event_id);
      if (!Number.isInteger(eventId) || eventId <= 0) return false;
      if (requireVisuallySimilar && !match?.visually_similar) return false;
      return true;
    })
    .map((match) => ({
      query_mode: "appearance",
      score: Number(match.similarity) || 0,
      rank_score: Number(match.similarity) || 0,
      match_strength: match.visually_similar ? "appearance_match" : "appearance_candidate",
      evidence: {
        source_kind: "appearance",
        object_label: String(match.candidate_label || match.anchor_label || ""),
        anchor_track_id: match.anchor_track_id ?? null,
        candidate_track_id: match.candidate_track_id ?? null,
      },
      event: {
        id: Number(match.event_id),
        camera_id: String(match.camera_id || ""),
        created_at: String(match.created_at || ""),
      },
      snapshot_url: "",
    }));
}

export function normalizeVisualFindSimilarHits(results) {
  return (Array.isArray(results) ? results : [])
    .map((result) => {
      const event = result?.event || {};
      const eventId = Number(event.id);
      if (!Number.isInteger(eventId) || eventId <= 0) return null;
      return {
        ...result,
        query_mode: "visual",
        event: {
          id: eventId,
          camera_id: String(event.camera_id || ""),
          created_at: String(event.created_at || ""),
          kind: event.kind,
          faces: event.faces,
        },
      };
    })
    .filter(Boolean);
}

export function mergeHybridFindSimilarResults({
  appearanceMatches = [],
  visualResults = [],
  limit = 16,
} = {}) {
  const bounded = Math.max(1, Math.min(48, Number(limit) || 16));
  const appearanceHits = normalizeAppearanceFindSimilarHits(appearanceMatches);
  const visualHits = normalizeVisualFindSimilarHits(visualResults);
  const merged = [];
  const seen = new Set();
  for (const hit of appearanceHits) {
    const eventId = Number(hit.event.id);
    if (seen.has(eventId)) continue;
    seen.add(eventId);
    merged.push(hit);
    if (merged.length >= bounded) return merged;
  }
  for (const hit of visualHits) {
    const eventId = Number(hit.event.id);
    if (seen.has(eventId)) continue;
    seen.add(eventId);
    merged.push(hit);
    if (merged.length >= bounded) break;
  }
  return merged;
}

export function hybridMatchLabel(result) {
  if (result?.query_mode === "appearance") {
    const similarity = Number(result.score);
    if (Number.isFinite(similarity)) return `Appearance ${Math.round(similarity * 100)}%`;
    return "Appearance match";
  }
  return visualMatchLabel(result?.match_strength);
}

export function hybridFindSimilarSubtitle({ objectLabel, usedAppearance, usedVisual, trackId }) {
  const label = String(objectLabel || "object").trim() || "object";
  if (usedAppearance && usedVisual) {
    return trackId
      ? `Appearance trail for ${label} #${trackId}, broadened with visual similarity`
      : `Appearance trail for ${label}, broadened with visual similarity`;
  }
  if (usedAppearance) {
    return trackId
      ? `Appearance matches for ${label} #${trackId}`
      : `Appearance matches for ${label}`;
  }
  return `Visually similar to ${label}`;
}
