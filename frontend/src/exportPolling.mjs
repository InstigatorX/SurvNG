export const ACTIVE_EXPORT_STATUSES = Object.freeze(["queued", "running", "cancelling"]);

const ACTIVE_EXPORT_STATUS_SET = new Set(ACTIVE_EXPORT_STATUSES);
const jobCache = new Map();
const jobRequests = new Map();

export function exportIsActive(job) {
  return Boolean(job?.id && ACTIVE_EXPORT_STATUS_SET.has(job.status));
}

export function assistantActiveExportIds(messages) {
  return [...new Set((messages || []).flatMap((message) =>
    (message.evidence || [])
      .map((item) => item.details?.media_export)
      .filter(exportIsActive)
      .map((job) => String(job.id))
  ))].sort();
}

export function cacheExportJobs(jobs, observedAt = Date.now()) {
  for (const job of jobs || []) {
    if (!job?.id) continue;
    jobCache.set(String(job.id), { job, observedAt });
  }
}

export function removeCachedExportJobs(ids) {
  for (const id of ids || []) jobCache.delete(String(id || ""));
}

export function cachedExportJob(id, maxAgeMs = Infinity, now = Date.now()) {
  const cached = jobCache.get(String(id || ""));
  if (!cached || now - cached.observedAt > Math.max(0, Number(maxAgeMs) || 0)) return null;
  return cached.job;
}

export async function fetchExportJob(id, request, { signal, maxAgeMs = 0 } = {}) {
  const normalizedId = String(id || "");
  if (!normalizedId) return null;
  const cached = cachedExportJob(normalizedId, maxAgeMs);
  if (cached) return cached;
  if (jobRequests.has(normalizedId)) return jobRequests.get(normalizedId);
  const pending = (async () => {
    const response = await request(`/api/exports/${encodeURIComponent(normalizedId)}`, { signal });
    if (!response.ok) throw new Error(`Export status failed (${response.status})`);
    const job = await response.json();
    cacheExportJobs([job]);
    return job;
  })();
  jobRequests.set(normalizedId, pending);
  try {
    return await pending;
  } finally {
    if (jobRequests.get(normalizedId) === pending) jobRequests.delete(normalizedId);
  }
}

export async function fetchAssistantExportJobs(ids, request, { signal } = {}) {
  const requestedIds = [...new Set((ids || []).map(String).filter(Boolean))];
  if (!requestedIds.length) return [];
  let activeJobs = [];
  try {
    const response = await request("/api/exports?status=active&limit=500", { signal });
    if (!response.ok) throw new Error(`Export list failed (${response.status})`);
    const payload = await response.json();
    activeJobs = Array.isArray(payload.exports) ? payload.exports : [];
    cacheExportJobs(activeJobs);
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    // Retain the older per-job behavior as a compatibility fallback when the
    // list endpoint is unavailable.
    return (await Promise.all(requestedIds.map((id) =>
      fetchExportJob(id, request, { signal, maxAgeMs: 750 }).catch(() => null)
    ))).filter(Boolean);
  }

  const activeById = new Map(activeJobs.map((job) => [String(job.id), job]));
  const resolved = requestedIds.map((id) => activeById.get(id)).filter(Boolean);
  const missingIds = requestedIds.filter((id) => !activeById.has(id));
  if (missingIds.length) {
    const terminal = await Promise.all(missingIds.map((id) =>
      fetchExportJob(id, request, { signal, maxAgeMs: 750 }).catch(() => null)
    ));
    resolved.push(...terminal.filter(Boolean));
  }
  return resolved;
}

export function mergeAssistantExportJobs(messages, jobs) {
  const byId = new Map((jobs || []).filter((job) => job?.id).map((job) => [String(job.id), job]));
  if (!byId.size) return messages;
  return (messages || []).map((message) => ({
    ...message,
    evidence: (message.evidence || []).map((item) => {
      const previous = item.details?.media_export;
      const update = previous?.id ? byId.get(String(previous.id)) : null;
      if (!update) return item;
      return {
        ...item,
        details: {
          ...item.details,
          media_export: {
            ...previous,
            status: update.status,
            phase: update.phase,
            progress: update.progress,
            error: update.error,
            output_name: update.output_name,
            size_bytes: update.size_bytes,
            download_url: update.download_url,
          },
        },
      };
    }),
  }));
}

export function resetExportPollingCacheForTests() {
  jobCache.clear();
  jobRequests.clear();
}
