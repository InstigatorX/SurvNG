const CSRF_EXEMPT_PATHS = new Set(["/api/auth/login", "/api/auth/bootstrap"]);
const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function normalizedBasePath(basePath) {
  return String(basePath || "").replace(/\/+$/, "");
}

function pathnameOnly(path) {
  const end = path.search(/[?#]/);
  return end === -1 ? path : path.slice(0, end);
}

/** Logical app path, with a configured base path removed when the caller already added it. */
export function requestApiPath(resource, basePath = "") {
  if (typeof resource !== "string" || !resource.startsWith("/") || resource.startsWith("//")) {
    return "";
  }
  const base = normalizedBasePath(basePath);
  if (base && (resource === base || resource.startsWith(`${base}/`))) {
    return resource.slice(base.length) || "/";
  }
  return resource;
}

export function mutationNeedsCsrf(resource, method, basePath = "") {
  const path = pathnameOnly(requestApiPath(resource, basePath));
  return (
    path.startsWith("/api/")
    && !CSRF_EXEMPT_PATHS.has(path)
    && UNSAFE_METHODS.has(String(method || "GET").toUpperCase())
  );
}
