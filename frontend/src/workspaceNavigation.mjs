export const WORKSPACES = Object.freeze([
  Object.freeze({ id: "live", label: "Live", path: "/", paths: ["/"], legacyRoutes: { "/live": "/" } }),
  Object.freeze({ id: "incidents", label: "Incidents", path: "/incidents", paths: ["/incidents"], legacyRoutes: {} }),
  Object.freeze({ id: "timeline", label: "Timeline", path: "/timeline", paths: ["/timeline", "/timeline/exports"], legacyRoutes: { "/recordings": "/timeline", "/recordings/exports": "/timeline/exports" } }),
  Object.freeze({ id: "search", label: "Search", path: "/search", paths: ["/search"], legacyRoutes: { "/recordings/search": "/search" } }),
  Object.freeze({ id: "people", label: "People", path: "/people", paths: ["/people"], legacyRoutes: { "/faces": "/people" } }),
  Object.freeze({ id: "admin", label: "Admin", path: "/admin", paths: ["/admin"], legacyRoutes: { "/config": "/admin" } }),
]);

export const DESKTOP_PRIMARY_WORKSPACES = Object.freeze([
  "live",
  "incidents",
  "timeline",
  "search",
  "people",
]);

export const MOBILE_PRIMARY_WORKSPACES = Object.freeze([
  "live",
  "incidents",
  "timeline",
  "search",
  "more",
]);

const WORKSPACE_BY_ID = new Map(WORKSPACES.map((workspace) => [workspace.id, workspace]));

function normalizedPath(pathname) {
  const path = String(pathname || "/").trim() || "/";
  if (!path.startsWith("/") || path.startsWith("//")) return "/";
  const withoutTrailingSlash = path.length > 1 ? path.replace(/\/+$/, "") : path;
  return withoutTrailingSlash || "/";
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

export function canonicalWorkspaceUrl(pathname, search = "", hash = "") {
  const safeSearch = String(search || "").startsWith("?") ? String(search) : search ? `?${search}` : "";
  const safeHash = String(hash || "").startsWith("#") ? String(hash) : hash ? `#${hash}` : "";
  return `${canonicalWorkspacePath(pathname)}${safeSearch}${safeHash}`;
}

export function workspaceHref(workspaceId, params = {}) {
  const workspace = workspaceDefinition(workspaceId);
  if (!workspace) throw new Error(`Unknown SurvNG workspace: ${workspaceId}`);
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    search.set(key, String(value));
  });
  return `${workspace.path}${search.size ? `?${search.toString()}` : ""}`;
}

export function timelineHref({ cameraId, epoch, source, date } = {}) {
  const params = {};
  if (cameraId) params.camera = cameraId;
  if (Number.isFinite(Number(epoch))) params.at = Number(epoch);
  if (source) params.source = source;
  if (date) params.date = date;
  return workspaceHref("timeline", params);
}

export function systemHealthState({ lifecycle, storage, detector, cameras } = {}) {
  if (lifecycle && lifecycle !== "running") {
    return { healthy: false, severity: "starting", label: String(lifecycle).replaceAll("_", " ") };
  }
  const knownFailure = storage?.available === false
    || (detector?.enabled !== false && detector != null && !detector.loaded_backend)
    || (Number(cameras?.enabled) > 0 && Number(cameras?.online) < Number(cameras.enabled))
    || (Number(cameras?.recording_expected) > 0 && Number(cameras?.recording) < Number(cameras.recording_expected));
  if (knownFailure) return { healthy: false, severity: "attention", label: "Needs attention" };
  const fullyKnown = lifecycle === "running"
    && storage?.available === true
    && detector != null
    && cameras != null;
  return fullyKnown
    ? { healthy: true, severity: "healthy", label: "Healthy" }
    : { healthy: false, severity: "checking", label: "Checking" };
}
