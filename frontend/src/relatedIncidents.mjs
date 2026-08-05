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
  if (expectedRoute && hasAppearance && hasTime) return `Expected route · Appearance ${Math.round(similarity * 100)}% · ${Math.round(seconds)}s`;
  if (expectedRoute && hasTime) return `Expected route · ${Math.round(seconds)}s`;
  if (hasAppearance && hasTime) return `Appearance ${Math.round(similarity * 100)}% · ${Math.round(seconds)}s`;
  if (hasAppearance) return `Appearance ${Math.round(similarity * 100)}%`;
  if (hasTime) return `Likely sequence · ${Math.round(seconds)}s`;
  return "Related incident";
}
