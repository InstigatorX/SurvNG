export const WORKSPACES = Object.freeze([
  Object.freeze({ id: "live", label: "Live", path: "/", legacyPaths: ["/live"] }),
  Object.freeze({ id: "incidents", label: "Incidents", path: "/incidents", legacyPaths: [] }),
  Object.freeze({ id: "timeline", label: "Timeline", path: "/timeline", legacyPaths: ["/recordings"] }),
  Object.freeze({ id: "search", label: "Search", path: "/search", legacyPaths: ["/recordings/search"] }),
  Object.freeze({ id: "people", label: "People", path: "/people", legacyPaths: ["/faces"] }),
  Object.freeze({ id: "admin", label: "Admin", path: "/admin", legacyPaths: ["/config"] }),
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

function pathMatches(candidate, root) {
  return candidate === root || (root !== "/" && candidate.startsWith(`${root}/`));
}

export function workspaceDefinition(workspaceId) {
  return WORKSPACE_BY_ID.get(workspaceId) || null;
}

export function resolveWorkspace(pathname) {
  const path = normalizedPath(pathname);
  const candidates = WORKSPACES.flatMap((workspace) => [workspace.path, ...workspace.legacyPaths]
    .filter((root) => root !== "/")
    .map((root) => ({ workspace, root })))
    .sort((left, right) => right.root.length - left.root.length);
  const matched = candidates.find(({ root }) => pathMatches(path, root));
  if (matched) return matched.workspace;
  return path === "/" ? WORKSPACE_BY_ID.get("live") : null;
}

export function canonicalWorkspacePath(pathname) {
  const path = normalizedPath(pathname);
  const aliases = WORKSPACES.flatMap((workspace) => workspace.legacyPaths
    .map((legacyRoot) => ({ workspace, legacyRoot })))
    .sort((left, right) => right.legacyRoot.length - left.legacyRoot.length);
  const matched = aliases.find(({ legacyRoot }) => pathMatches(path, legacyRoot));
  if (matched) {
    const suffix = path.slice(matched.legacyRoot.length);
    return `${matched.workspace.path}${suffix}` || "/";
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
