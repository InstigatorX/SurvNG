export const SEMANTIC_SEARCH_HISTORY_KEY = "survng.semanticSearchHistory.v1";
export const SEMANTIC_SEARCH_SESSION_KEY = "survng.semanticSearchSession.v1";

export function semanticSearchResultsForCamera(results, cameraId) {
  const items = Array.isArray(results) ? results : [];
  const selectedCameraId = String(cameraId || "");
  return selectedCameraId
    ? items.filter((result) => String(result?.event?.camera_id || "") === selectedCameraId)
    : items;
}

function normalizedEntry(value) {
  const query = String(value?.query || "").trim();
  if (!query) return null;
  return {
    query,
    cameraId: String(value?.cameraId || "").trim(),
    searchedAt: String(value?.searchedAt || ""),
  };
}

export function readSemanticSearchHistory(storage) {
  try {
    const parsed = JSON.parse(storage?.getItem(SEMANTIC_SEARCH_HISTORY_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizedEntry).filter(Boolean).slice(0, 5);
  } catch {
    return [];
  }
}

export function addSemanticSearchHistory(history, value) {
  const entry = normalizedEntry(value);
  if (!entry) return Array.isArray(history) ? history.slice(0, 5) : [];
  const identity = `${entry.query.toLocaleLowerCase()}\u0000${entry.cameraId}`;
  return [
    entry,
    ...(Array.isArray(history) ? history : []).filter((candidate) => {
      const normalized = normalizedEntry(candidate);
      return normalized
        && `${normalized.query.toLocaleLowerCase()}\u0000${normalized.cameraId}` !== identity;
    }),
  ].slice(0, 5);
}

export function writeSemanticSearchHistory(storage, history) {
  try {
    storage?.setItem(SEMANTIC_SEARCH_HISTORY_KEY, JSON.stringify((history || []).slice(0, 5)));
    return true;
  } catch {
    return false;
  }
}

export function readSemanticSearchSession(storage, query, cameraId) {
  const expectedQuery = String(query || "").trim();
  if (!expectedQuery) return null;
  try {
    const parsed = JSON.parse(storage?.getItem(SEMANTIC_SEARCH_SESSION_KEY) || "null");
    if (!parsed || String(parsed.query || "").trim() !== expectedQuery) return null;
    if (String(parsed.cameraId || "") !== String(cameraId || "")) return null;
    if (!Array.isArray(parsed.results)) return null;
    return { query: expectedQuery, cameraId: String(cameraId || ""), results: parsed.results };
  } catch {
    return null;
  }
}

export function writeSemanticSearchSession(storage, value) {
  const query = String(value?.query || "").trim();
  if (!query || !Array.isArray(value?.results)) return false;
  try {
    storage?.setItem(SEMANTIC_SEARCH_SESSION_KEY, JSON.stringify({
      query,
      cameraId: String(value.cameraId || ""),
      results: value.results,
    }));
    return true;
  } catch {
    return false;
  }
}

export function clearSemanticSearchSession(storage) {
  try {
    storage?.removeItem(SEMANTIC_SEARCH_SESSION_KEY);
    return true;
  } catch {
    return false;
  }
}
