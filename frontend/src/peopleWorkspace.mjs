export const PEOPLE_REVIEW_FILTERS = Object.freeze({
  unknown: "Needs review",
  suggested: "Suggestions",
  known: "Confirmed",
  pending: "Processing",
  unusable: "Unusable",
  all: "All",
});

export const PEOPLE_WORKSPACE_MODES = Object.freeze({
  review: "Review queue",
  people: "People profiles",
  clusters: "Unknown clusters",
});

export function readPeopleWorkspaceQuery(search = "") {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  const requestedStatus = String(params.get("status") || "unknown");
  const requestedMode = String(params.get("mode") || "review");
  const pageNumber = Number(params.get("page") || 1);
  return {
    mode: Object.hasOwn(PEOPLE_WORKSPACE_MODES, requestedMode) ? requestedMode : "review",
    status: Object.hasOwn(PEOPLE_REVIEW_FILTERS, requestedStatus) ? requestedStatus : "unknown",
    cameraId: String(params.get("camera") || ""),
    personId: String(params.get("person") || ""),
    page: Number.isFinite(pageNumber) ? Math.max(0, Math.floor(pageNumber) - 1) : 0,
    faceId: String(params.get("face") || ""),
    clusterId: String(params.get("cluster") || ""),
  };
}

export function peopleWorkspaceSearch({ mode = "review", status = "unknown", cameraId = "", personId = "", page = 0, faceId = "", clusterId = "" } = {}) {
  const params = new URLSearchParams();
  if (mode && mode !== "review") params.set("mode", mode);
  if (status && status !== "unknown") params.set("status", status);
  if (cameraId) params.set("camera", cameraId);
  if (personId) params.set("person", personId);
  if (Number(page) > 0) params.set("page", String(Math.floor(Number(page)) + 1));
  if (faceId) params.set("face", faceId);
  if (clusterId) params.set("cluster", clusterId);
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function peopleFilterLabel(value) {
  return PEOPLE_REVIEW_FILTERS[value] || PEOPLE_REVIEW_FILTERS.unknown;
}

export function peopleModeLabel(value) {
  return PEOPLE_WORKSPACE_MODES[value] || PEOPLE_WORKSPACE_MODES.review;
}

export function peopleObservationRequestPlan({
  mode = "review",
  status = "unknown",
  cameraId = "",
  personId = "",
  page = 0,
  pageSize = 48,
} = {}) {
  if (mode === "clusters") return { observations: "", count: "" };
  const limit = Math.max(1, Math.floor(Number(pageSize) || 48));
  if (mode === "review") {
    return { observations: `/api/faces/review/queue?limit=${limit}`, count: "" };
  }
  const query = new URLSearchParams({
    status: personId ? "all" : status,
    limit: String(limit),
    offset: String(Math.max(0, Math.floor(Number(page) || 0)) * limit),
  });
  if (cameraId) query.set("camera_id", cameraId);
  if (personId) query.set("person_id", personId);
  const countQuery = new URLSearchParams(query);
  countQuery.delete("limit");
  countQuery.delete("offset");
  return {
    observations: `/api/faces/observations?${query}`,
    count: `/api/faces/observations/count?${countQuery}`,
  };
}
