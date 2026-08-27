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
