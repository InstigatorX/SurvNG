/**
 * Comma-separated event types the current listeners read, or "" for all.
 * The server then skips the rest (e.g. motion and object traffic on Incidents).
 */
export function appEventTypesKey(typeLists) {
  const union = new Set();
  for (const types of typeLists) {
    if (!types) return "";
    types.forEach((type) => union.add(type));
  }
  return [...union].sort().join(",");
}
