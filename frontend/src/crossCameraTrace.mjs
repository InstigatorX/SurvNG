export function crossCameraTracePath(anchorEventId, params = {}) {
  const eventId = Number(anchorEventId);
  const search = new URLSearchParams();
  if (params.start_at) search.set("start_at", String(params.start_at));
  if (params.end_at) search.set("end_at", String(params.end_at));
  if (params.object_label) search.set("object_label", String(params.object_label));
  if (params.face_name) search.set("face_name", String(params.face_name));
  if (params.time_zone) search.set("time_zone", String(params.time_zone));
  if (params.limit) search.set("limit", String(Math.max(1, Math.min(12, Number(params.limit) || 12))));
  const query = search.toString();
  return `/api/incidents/by-event/${eventId}/cross-camera-trace${query ? `?${query}` : ""}`;
}

const MATCH_STRENGTH_LABELS = {
  confirmed_identity: "Confirmed face",
  automatic_identity: "Automatic face match",
  possible_identity: "Possible face",
  appearance_similarity: "Visually similar appearance",
  context_candidate: "Nearby matching class",
};

export function crossCameraMatchLabel(match) {
  const strength = String(match?.match_strength || "");
  if (strength === "appearance_similarity") {
    const similarity = Number(match?.appearance_similarity);
    return Number.isFinite(similarity)
      ? `Visually similar ${Math.round(similarity * 100)}%`
      : MATCH_STRENGTH_LABELS.appearance_similarity;
  }
  return MATCH_STRENGTH_LABELS[strength] || "Possible connection";
}

export function crossCameraMatchCameraLabel(match, cameraNameById) {
  const cameraId = String(match?.camera_id || "");
  return cameraNameById?.get?.(cameraId) || cameraId || "Unknown camera";
}
