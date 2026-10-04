const ACTIVE = new Set(["queued", "running"]);
const TERMINAL = new Set(["complete", "partial", "unavailable"]);

// Share admission/renewal across open surfaces and React's development remounts.
// This is only a short browser-side debounce; the durable job owns deduplication.
export function createAnalysisRequester(fetch, now = Date.now) {
  const requests = new Map();
  return function request(incidentId) {
    const previous = requests.get(incidentId);
    if (previous && now() - previous.at < 30000) return previous.promise;
    const entry = { at: now() };
    entry.promise = (async () => {
      const response = await fetch(`/api/incidents/${encodeURIComponent(incidentId)}/analysis`, { method: "POST" });
      if (!response.ok) throw new Error("Could not request extra incident details.");
      return response.json();
    })().catch((error) => {
      if (requests.get(incidentId) === entry) requests.delete(incidentId);
      throw error;
    });
    requests.set(incidentId, entry);
    if (requests.size > 64) requests.delete(requests.keys().next().value);
    return entry.promise;
  };
}

export function analysisLabel(status) {
  return ({ deferred: "Extra details waiting", queued: "Extra details queued", running: "Analyzing extra details", complete: "Extra details ready", partial: "Extra details partly available", unavailable: "Extra details unavailable" })[status] || "Extra incident details";
}

// Kept independent of React so visibility, expiry, retries and teardown can be
// tested with a clock, without waiting for real 30-second renewals.
export function createIncidentAnalysisViewer({ incidentId, fetch, request, onChange, onProgress, visible, autoStart = true, now = Date.now }) {
  let disposed = false, busy = false, terminal = false, admissionFailed = false;
  let activated = autoStart;
  let nextPollAt = 0, lastAdmissionAt = -Infinity;
  const controller = new AbortController();
  async function poll({ force = false } = {}) {
    if (disposed || busy || !visible() || (!force && (terminal || now() < nextPollAt))) return;
    busy = true;
    try {
      const response = await fetch(`/api/incidents/${encodeURIComponent(incidentId)}/analysis`, { signal: controller.signal });
      if (disposed || !visible()) return;
      // Older servers and deleted incidents do not support this optional panel.
      if (response.status === 404) { terminal = true; onChange(null); return; }
      if (!response.ok) throw new Error("Extra details status is temporarily unavailable.");
      let data = await response.json();
      if (disposed || !visible()) return;
      if (activated && data.enabled && !admissionFailed && (data.status === "deferred" || ACTIVE.has(data.status)) && now() - lastAdmissionAt >= 30000) {
        lastAdmissionAt = now();
        try { data = await request(incidentId); }
        catch (error) { admissionFailed = true; throw error; }
      }
      if (disposed || !visible()) return;
      terminal = !data.enabled || TERMINAL.has(data.status) || (!activated && data.status === "deferred");
      onChange({ ...data, error: "" });
      if (data.enabled && data.status !== "deferred") onProgress?.(data);
      nextPollAt = now() + 5000;
    } catch (error) {
      if (disposed || error.name === "AbortError") return;
      onChange({ error: `${error.message} Video and saved evidence are still available.` });
      nextPollAt = now() + 30000;
    } finally { busy = false; }
  }
  return {
    poll,
    start() { activated = true; return poll({ force: true }); },
    retry() { admissionFailed = false; lastAdmissionAt = -Infinity; return poll({ force: true }); },
    dispose() { disposed = true; controller.abort(); },
  };
}
