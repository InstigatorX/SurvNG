export const VISUAL_SEARCH_TRAIL_KEY = "survng.visualSearchTrail.v1";
export const MAX_TRAIL_EVENTS = 24;

export function normalizeTrailEventIds(values, limit = MAX_TRAIL_EVENTS) {
  const bounded = Math.max(1, Math.min(MAX_TRAIL_EVENTS, Number(limit) || MAX_TRAIL_EVENTS));
  const seen = new Set();
  const ids = [];
  for (const value of Array.isArray(values) ? values : []) {
    const eventId = Number(value);
    if (!Number.isInteger(eventId) || eventId <= 0 || seen.has(eventId)) continue;
    seen.add(eventId);
    ids.push(eventId);
    if (ids.length >= bounded) break;
  }
  return ids;
}

export function serializeTrailEventIds(values) {
  const ids = normalizeTrailEventIds(values);
  return ids.length ? ids.join(",") : "";
}

export function parseTrailEventIds(raw) {
  if (Array.isArray(raw)) return normalizeTrailEventIds(raw);
  return normalizeTrailEventIds(String(raw || "").split(","));
}

export function trailPosition(eventIds, eventId) {
  const ids = normalizeTrailEventIds(eventIds);
  const current = Number(eventId);
  if (!ids.length || !Number.isInteger(current) || current <= 0) {
    return { index: -1, count: ids.length, previousId: null, nextId: null };
  }
  const index = ids.indexOf(current);
  return {
    index,
    count: ids.length,
    previousId: index > 0 ? ids[index - 1] : null,
    nextId: index >= 0 && index < ids.length - 1 ? ids[index + 1] : null,
  };
}

export function normalizeTrailHit(value) {
  const event = value?.event || {};
  const eventId = Number(event.id || value?.event_id);
  if (!Number.isInteger(eventId) || eventId <= 0) return null;
  const createdAt = String(event.created_at || value?.created_at || "");
  const epoch = Number.isFinite(Number(event.incident_epoch))
    ? Number(event.incident_epoch)
    : (createdAt ? new Date(createdAt).getTime() / 1000 : NaN);
  const snapshotPath = String(event.snapshot_path || value?.snapshot_path || "").trim();
  return {
    query_mode: value?.query_mode === "appearance" ? "appearance" : "visual",
    event: {
      id: eventId,
      camera_id: String(event.camera_id || value?.camera_id || ""),
      created_at: createdAt,
      incident_epoch: Number.isFinite(epoch) ? epoch : null,
      kind: event.kind,
      labels: Array.isArray(event.labels) ? event.labels : undefined,
      snapshot_path: snapshotPath || "available",
    },
  };
}

export function writeVisualSearchTrail(storage, {
  eventIds = [],
  hits = [],
  queryMode = null,
} = {}) {
  const normalizedIds = normalizeTrailEventIds(eventIds);
  const normalizedHits = (Array.isArray(hits) ? hits : [])
    .map(normalizeTrailHit)
    .filter(Boolean)
    .filter((hit) => normalizedIds.includes(hit.event.id));
  const payload = {
    eventIds: normalizedIds,
    queryMode: queryMode === "appearance" || queryMode === "visual" ? queryMode : null,
    hits: normalizedHits,
    savedAt: new Date().toISOString(),
  };
  try {
    storage?.setItem(VISUAL_SEARCH_TRAIL_KEY, JSON.stringify(payload));
    return payload;
  } catch {
    return payload;
  }
}

export function readVisualSearchTrail(storage) {
  try {
    const parsed = JSON.parse(storage?.getItem(VISUAL_SEARCH_TRAIL_KEY) || "null");
    if (!parsed || typeof parsed !== "object") return null;
    const eventIds = normalizeTrailEventIds(parsed.eventIds);
    if (!eventIds.length) return null;
    return {
      eventIds,
      queryMode: parsed.queryMode === "appearance" || parsed.queryMode === "visual"
        ? parsed.queryMode
        : null,
      hits: (Array.isArray(parsed.hits) ? parsed.hits : [])
        .map(normalizeTrailHit)
        .filter(Boolean)
        .filter((hit) => eventIds.includes(hit.event.id)),
      savedAt: String(parsed.savedAt || ""),
    };
  } catch {
    return null;
  }
}

export function clearVisualSearchTrail(storage) {
  try {
    storage?.removeItem(VISUAL_SEARCH_TRAIL_KEY);
  } catch {
    // Ignore quota / private-mode failures.
  }
}

export function trailHitForEvent(trail, eventId) {
  const current = Number(eventId);
  if (!trail || !Number.isInteger(current) || current <= 0) return null;
  return (trail.hits || []).find((hit) => Number(hit.event.id) === current) || null;
}
