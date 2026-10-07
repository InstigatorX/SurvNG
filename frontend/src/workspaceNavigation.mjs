export const REVIEW_MODES = Object.freeze(["incidents", "timeline", "search"]);

const REVIEW_LEGACY_MODES = Object.freeze({
  "/incidents": "incidents",
  "/timeline": "timeline",
  "/recordings": "timeline",
  "/search": "search",
  "/recordings/search": "search",
});

export const WORKSPACES = Object.freeze([
  Object.freeze({ id: "observations", label: "Observations", path: "/observations", paths: ["/observations"], legacyRoutes: {} }),
  Object.freeze({ id: "live", label: "Live", path: "/", paths: ["/"], legacyRoutes: { "/live": "/" } }),
  Object.freeze({
    id: "review",
    label: "Review",
    path: "/review",
    paths: ["/review"],
    legacyRoutes: {
      "/incidents": "/review",
      "/timeline": "/review",
      "/recordings": "/review",
      "/search": "/review",
      "/recordings/search": "/review",
    },
  }),
  Object.freeze({ id: "exports", label: "Exports", path: "/exports", paths: ["/exports"], legacyRoutes: { "/timeline/exports": "/exports", "/recordings/exports": "/exports" } }),
  Object.freeze({ id: "people", label: "People", path: "/people", paths: ["/people"], legacyRoutes: { "/faces": "/people" } }),
  Object.freeze({ id: "admin", label: "System", path: "/admin", paths: ["/admin"], legacyRoutes: { "/config": "/admin", "/system": "/admin" } }),
]);

export const DESKTOP_PRIMARY_WORKSPACES = Object.freeze([
  "live",
  "review",
  "people",
]);

export const MOBILE_PRIMARY_WORKSPACES = Object.freeze([
  "live",
  "review",
  "people",
]);

const WORKSPACE_BY_ID = new Map(WORKSPACES.map((workspace) => [workspace.id, workspace]));

function normalizedPath(pathname) {
  const path = String(pathname || "/").trim() || "/";
  if (!path.startsWith("/") || path.startsWith("//")) return "/";
  const withoutTrailingSlash = path.length > 1 ? path.replace(/\/+$/, "") : path;
  return withoutTrailingSlash || "/";
}

function searchParams(search = "") {
  const value = String(search || "");
  return new URLSearchParams(value.startsWith("?") ? value.slice(1) : value);
}

export function workspaceDefinition(workspaceId) {
  return WORKSPACE_BY_ID.get(workspaceId) || null;
}

export function resolveWorkspace(pathname) {
  const path = normalizedPath(pathname);
  const matched = WORKSPACES.find((workspace) => (
    workspace.paths.includes(path) || Object.hasOwn(workspace.legacyRoutes, path)
  ));
  if (matched) return matched;
  return null;
}

export function canonicalWorkspacePath(pathname) {
  const path = normalizedPath(pathname);
  for (const workspace of WORKSPACES) {
    if (Object.hasOwn(workspace.legacyRoutes, path)) return workspace.legacyRoutes[path];
  }
  return path;
}

export function reviewMode(pathname, search = "") {
  const path = canonicalWorkspacePath(pathname);
  const source = normalizedPath(pathname);
  if (path !== "/review" && source !== "/review") return null;
  const explicit = searchParams(search).get("mode");
  if (REVIEW_MODES.includes(explicit)) return explicit;
  return REVIEW_LEGACY_MODES[source] || "incidents";
}

export function canonicalWorkspaceUrl(pathname, search = "", hash = "") {
  const source = normalizedPath(pathname);
  const canonicalPath = canonicalWorkspacePath(pathname);
  const params = searchParams(search);
  const legacyMode = REVIEW_LEGACY_MODES[source];
  if (canonicalPath === "/review" && legacyMode && legacyMode !== "incidents" && !params.get("mode")) {
    params.set("mode", legacyMode);
  }
  if (canonicalPath === "/review" && params.get("mode") === "incidents") params.delete("mode");
  const query = params.toString();
  const safeHash = String(hash || "").startsWith("#") ? String(hash) : hash ? `#${hash}` : "";
  return `${canonicalPath}${query ? `?${query}` : ""}${safeHash}`;
}

export function workspaceHref(workspaceId, params = {}) {
  const workspace = workspaceDefinition(workspaceId);
  if (!workspace) throw new Error(`Unknown SurvNG workspace: ${workspaceId}`);
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    search.set(key, String(value));
  });
  if (workspaceId === "review" && search.get("mode") === "incidents") search.delete("mode");
  return `${workspace.path}${search.size ? `?${search.toString()}` : ""}`;
}

export function timelineHref({
  cameraId,
  epoch,
  source,
  date,
  eventId,
  trailEventIds,
  queryMode,
} = {}) {
  const params = { mode: "timeline" };
  if (cameraId) params.camera = cameraId;
  if (Number.isFinite(Number(epoch))) params.at = Number(epoch);
  if (source) params.source = source;
  if (date) params.date = date;
  if (Number.isInteger(Number(eventId)) && Number(eventId) > 0) params.event = Number(eventId);
  const trail = Array.isArray(trailEventIds)
    ? trailEventIds.filter((id) => Number.isInteger(Number(id)) && Number(id) > 0)
    : [];
  if (trail.length) {
    const seen = new Set();
    params.trail = trail
      .map((id) => Number(id))
      .filter((id) => {
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      })
      .slice(0, 24)
      .join(",");
  }
  if (queryMode === "appearance" || queryMode === "visual") params.query_mode = queryMode;
  return workspaceHref("review", params);
}
