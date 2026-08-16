export const PEOPLE_REVIEW_FILTERS = Object.freeze({
  unknown: "Needs review",
  suggested: "Suggestions",
  known: "Confirmed",
  pending: "Processing",
  unusable: "Unusable",
  all: "All",
});

export function readPeopleWorkspaceQuery(search = "") {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  const requestedStatus = String(params.get("status") || "unknown");
  const pageNumber = Number(params.get("page") || 1);
  return {
    status: Object.hasOwn(PEOPLE_REVIEW_FILTERS, requestedStatus) ? requestedStatus : "unknown",
    cameraId: String(params.get("camera") || ""),
    personId: String(params.get("person") || ""),
    page: Number.isFinite(pageNumber) ? Math.max(0, Math.floor(pageNumber) - 1) : 0,
    faceId: String(params.get("face") || ""),
  };
}

export function peopleWorkspaceSearch({ status = "unknown", cameraId = "", personId = "", page = 0, faceId = "" } = {}) {
  const params = new URLSearchParams();
  if (status && status !== "unknown") params.set("status", status);
  if (cameraId) params.set("camera", cameraId);
  if (personId) params.set("person", personId);
  if (Number(page) > 0) params.set("page", String(Math.floor(Number(page)) + 1));
  if (faceId) params.set("face", faceId);
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function peopleFilterLabel(value) {
  return PEOPLE_REVIEW_FILTERS[value] || PEOPLE_REVIEW_FILTERS.unknown;
}

