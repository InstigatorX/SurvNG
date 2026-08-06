export function semanticIncidentRequest({ query, cameraFilter, objectFilter, startAt, endAt, limit = 60 }) {
  return {
    query: String(query || "").trim(),
    camera_ids: cameraFilter && cameraFilter !== "all" ? [cameraFilter] : [],
    object_labels: objectFilter && objectFilter !== "all" ? [objectFilter] : [],
    start_at: String(startAt || ""),
    end_at: String(endAt || ""),
    limit: Math.max(1, Math.min(100, Number(limit) || 60)),
  };
}

export function rankSemanticIncidentDetails(details, zoneFilter = "all") {
  const unique = new Map();
  for (const detail of Array.isArray(details) ? details : []) {
    if (!detail?.id) continue;
    const zones = Array.isArray(detail.zones) ? detail.zones : [];
    if (zoneFilter && zoneFilter !== "all" && !zones.includes(zoneFilter)) continue;
    const current = unique.get(detail.id);
    if (!current || Number(detail.semantic_search?.score || -1) > Number(current.semantic_search?.score || -1)) {
      unique.set(detail.id, detail);
    }
  }
  return Array.from(unique.values()).sort(
    (left, right) => Number(right.semantic_search?.score || -1) - Number(left.semantic_search?.score || -1),
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
