import { incidentFilterValues } from "./incidentNavigation.mjs";

export function semanticIncidentRequest({ query, cameraFilter, objectFilter, startAt, endAt, limit = 60 }) {
  return {
    query: String(query || "").trim(),
    camera_ids: incidentFilterValues(cameraFilter),
    object_labels: incidentFilterValues(objectFilter),
    start_at: String(startAt || ""),
    end_at: String(endAt || ""),
    limit: Math.max(1, Math.min(100, Number(limit) || 60)),
  };
}

// The highest detector confidence an incident carries for the chosen object
// types (any type when none is chosen), or null when it carries none.
export function incidentPeakConfidence(detail, labels = []) {
  const wanted = new Set(labels);
  const objects = [
    ...(Array.isArray(detail?.objects) ? detail.objects : []),
    ...(Array.isArray(detail?.scene_objects) ? detail.scene_objects.flatMap((object) => [object, ...(object.observations || []).map((item) => ({ ...item, label: item.label || object.label }))]) : []),
    ...(Array.isArray(detail?.events) ? detail.events.flatMap((event) => (Array.isArray(event.objects) ? event.objects : [])) : []),
  ];
  let peak = null;
  for (const object of objects) {
    const confidence = Number(object?.confidence);
    if (!Number.isFinite(confidence) || (wanted.size && !wanted.has(object.label))) continue;
    peak = peak == null ? confidence : Math.max(peak, confidence);
  }
  return peak;
}

export function rankSemanticIncidentDetails(details, zoneFilter = "all", { minConfidence = 0, labels = [] } = {}) {
  const semanticRank = (detail) => Number(
    detail?.semantic_search?.rank_score ?? detail?.semantic_search?.score ?? -1,
  );
  const unique = new Map();
  for (const detail of Array.isArray(details) ? details : []) {
    if (!detail?.id) continue;
    const zones = Array.isArray(detail.zones) ? detail.zones : [];
    const wantedZones = incidentFilterValues(zoneFilter);
    if (wantedZones.length && !wantedZones.some((zone) => zones.includes(zone))) continue;
    // The same rule as the list: one detection of a chosen type must reach the floor.
    if (minConfidence > 0 && !((incidentPeakConfidence(detail, labels) ?? -1) >= minConfidence)) continue;
    const current = unique.get(detail.id);
    if (!current || semanticRank(detail) > semanticRank(current)) {
      unique.set(detail.id, detail);
    }
  }
  return Array.from(unique.values()).sort(
    (left, right) => semanticRank(right) - semanticRank(left),
  );
}

export async function mapWithConcurrency(items, concurrency, mapper) {
  const values = Array.from(items || []);
  if (!values.length) return [];
  const results = new Array(values.length);
  let nextIndex = 0;
  const workerCount = Math.max(1, Math.min(values.length, Math.floor(Number(concurrency) || 1)));
  async function worker() {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(values[index], index);
    }
  }
  await Promise.all(Array.from({ length: workerCount }, worker));
  return results;
}

// Keep the matches the incident list would also show under the same review and
// people filters, and give each its review mark. ``listed`` maps the id of
// every incident the list returns to its verdict.
export function keepListedIncidents(details, listed) {
  return (Array.isArray(details) ? details : []).flatMap((detail) => {
    const id = String(detail?.incident_id || detail?.id || "");
    return id && listed.has(id) ? [{ ...detail, review_verdict: listed.get(id) }] : [];
  });
}
