export function recordingReviewMinute(epoch) {
  if (epoch == null || !Number.isFinite(Number(epoch)) || Number(epoch) < 0) return null;
  return Math.floor(Number(epoch) / 60) * 60;
}

export function recordingReviewClosed(start, now = Date.now() / 1000) {
  return start !== null && Number.isFinite(start) && start + 60 <= now - 5;
}

export function recordingReviewUrl(cameraId, source, start) {
  const params = new URLSearchParams({ epoch: String(start), source });
  return `/api/cameras/${encodeURIComponent(cameraId)}/recordings/review?${params}`;
}

export function recordingReviewActive(state) {
  return state === "queued" || state === "analyzing";
}

export function recordingReviewSummary(result) {
  const labels = {
    unreviewed: "Unreviewed", queued: "Queued", analyzing: "Analyzing sampled frames",
    sampled: "Sampled", partial: "Partial review", unavailable: "Footage unavailable", failed: "Review failed",
  };
  return labels[result?.state] || "Checking saved review…";
}
