export async function loadRelatedIncident(eventId, fetch) {
  const response = await fetch(`/api/incidents/by-event/${eventId}`);
  if (response.ok) return response.json();
  if (response.status !== 404) throw new Error("Could not load related incident. Please try again.");
  // Related matches include evidence that has not joined a canonical scene.
  const evidence = await fetch(`/api/events/${eventId}`);
  if (!evidence.ok) throw new Error(evidence.status === 404
    ? "This related event is no longer available."
    : "Could not load related evidence. Please try again.");
  const event = await evidence.json();
  return { ...event, representative_event_id: event.id, events: [event] };
}

export function relatedIncidentsPath(anchorEventId, hours = 24, limit = 16) {
  const eventId = Number(anchorEventId);
  const boundedHours = Math.max(1, Math.min(168, Number(hours) || 24));
  const boundedLimit = Math.max(1, Math.min(100, Number(limit) || 16));
  return `/api/events/${eventId}/related-incidents?hours=${boundedHours}&limit=${boundedLimit}`;
}

export function relatedIncidentThumbnailPath(eventId, width = 360, quality = 80) {
  const boundedWidth = Math.max(64, Math.min(1920, Number(width) || 360));
  const boundedQuality = Math.max(1, Math.min(100, Number(quality) || 80));
  return `/api/events/${Number(eventId)}/thumbnail.jpg?width=${boundedWidth}&quality=${boundedQuality}`;
}

export function visibleRelatedAppearances(payload, anchorEventId, limit = 8) {
  const anchor = Number(anchorEventId);
  const boundedLimit = Math.max(1, Math.min(24, Number(limit) || 8));
  const seen = new Set();
  return (Array.isArray(payload?.matches) ? payload.matches : [])
    .filter((match) => {
      const eventId = Number(match?.event_id);
      const relation = String(match?.relation_type || "");
      const visibleRelation = ["sequence_candidate", "expected_route", "appearance_route"].includes(relation);
      if ((!match?.visually_similar && !visibleRelation) || !Number.isInteger(eventId) || eventId <= 0 || eventId === anchor || seen.has(eventId)) return false;
      seen.add(eventId);
      return true;
    })
    .sort((left, right) => {
      const rank = (item) => item.relation_type === "appearance_route" ? 0
        : item.relation_type === "appearance_sequence" ? 1
          : item.relation_type === "expected_route" ? 2
            : (item.relation_type === "appearance" || item.visually_similar) ? 3 : 4;
      return rank(left) - rank(right)
        || Number(left.sequence_delta_seconds ?? Number.MAX_SAFE_INTEGER) - Number(right.sequence_delta_seconds ?? Number.MAX_SAFE_INTEGER)
        || Number(right.similarity || 0) - Number(left.similarity || 0);
    })
    .slice(0, boundedLimit);
}

export function relatedEvidenceLabel(match) {
  const seconds = Number(match?.sequence_delta_seconds);
  const similarity = Number(match?.similarity);
  const hasTime = Number.isFinite(seconds);
  const hasAppearance = Boolean(match?.visually_similar) && Number.isFinite(similarity);
  const expectedRoute = ["appearance_route", "expected_route"].includes(String(match?.relation_type || ""));
  if (expectedRoute && hasAppearance && hasTime) return `Expected · Appearance ${Math.round(similarity * 100)}% · ${Math.round(seconds)}s`;
  if (expectedRoute && hasTime) return `Expected · ${Math.round(seconds)}s`;
  if (hasAppearance && hasTime) return `Appearance ${Math.round(similarity * 100)}% · ${Math.round(seconds)}s`;
  if (hasAppearance) return `Appearance ${Math.round(similarity * 100)}%`;
  if (hasTime) return `Likely · ${Math.round(seconds)}s`;
  return "Related incident";
}

export function relatedEvidenceReasons(match) {
  const relation = String(match?.relation_type || "");
  const reasons = [];
  if (match?.visually_similar || ["appearance", "appearance_route", "appearance_sequence"].includes(relation)) {
    const similarity = match?.similarity;
    const score = similarity != null && Number.isFinite(Number(similarity))
      ? ` · ${Math.round(Number(similarity) * 100)}% similarity` : "";
    reasons.push({ kind: "appearance", label: `Appearance match (ReID)${score}` });
  }
  if (["expected_route", "appearance_route"].includes(relation)) {
    reasons.push({ kind: "route", label: `Expected camera route${match?.route_name ? ` · ${match.route_name}` : ""}` });
  }
  if (["sequence_candidate", "appearance_sequence", "expected_route", "appearance_route"].includes(relation)) {
    const seconds = match?.sequence_delta_seconds;
    const gap = seconds != null && Number.isFinite(Number(seconds))
      ? ` · ${Math.round(Number(seconds))}s apart` : "";
    reasons.push({ kind: "time", label: `Nearby in time${gap}` });
  }
  return reasons.length ? reasons : [{ kind: "related", label: "Related incident" }];
}
