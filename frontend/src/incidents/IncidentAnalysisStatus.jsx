import React, { useEffect, useRef, useState } from "react";
import { fetch } from "../shared/api.js";
import { analysisLabel, createAnalysisRequester, createIncidentAnalysisViewer } from "../incidentAnalysis.mjs";
import "./incident-analysis.css";

const request = createAnalysisRequester(fetch);

// Only mount in an opened incident, never in feed cards or prefetch paths.
export function IncidentAnalysisStatus({ incidentId, autoStart = true, onDetail }) {
  const [state, setState] = useState(null);
  const viewer = useRef(null);
  const detailHandler = useRef(onDetail);
  detailHandler.current = onDetail;
  useEffect(() => {
    setState(null);
    if (!incidentId) return undefined;
    let alive = true, detailBusy = false;
    const controller = new AbortController();
    const current = createIncidentAnalysisViewer({ incidentId, fetch, request, autoStart, onChange: setState, onProgress: async (data) => {
      if (!detailHandler.current || detailBusy || !["running", "complete", "partial"].includes(data.status)) return;
      detailBusy = true;
      try {
        const response = await fetch(`/api/incidents/${encodeURIComponent(incidentId)}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Could not refresh the added details. Reopen this incident to try again.");
        const detail = await response.json();
        if (alive && !document.hidden) detailHandler.current?.(detail);
      } catch (error) {
        if (alive && error.name !== "AbortError") setState((previous) => ({ ...previous, error: "Could not refresh the added details. Recording playback remains available." }));
      } finally { detailBusy = false; }
    }, visible: () => !document.hidden });
    viewer.current = current;
    const refresh = () => { void current.poll({ force: true }); };
    void current.poll();
    const timer = window.setInterval(() => { void current.poll(); }, 5000);
    document.addEventListener("visibilitychange", refresh);
    window.addEventListener("online", refresh);
    return () => {
      current.dispose();
      alive = false;
      controller.abort();
      viewer.current = null;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
      window.removeEventListener("online", refresh);
    };
  }, [incidentId, autoStart]);
  if (!state || (!state.error && state.mode !== "on_demand" && !state.enabled)) return null;
  return <aside className="incident-analysis-status" aria-label="Extra incident details">
    <span role="status"><strong>{state.error ? "Extra details paused" : analysisLabel(state.status)}</strong>{" "}{state.error || state.message || (state.status === "running" || state.status === "queued" ? "You can play the recording while details are prepared." : state.status === "partial" ? "Some footage could not be analyzed. This does not mean nothing happened." : state.status === "unavailable" ? "Saved evidence and any retained recording remain available." : "Recording playback does not wait for this analysis.")}</span>
    {state.error ? <button type="button" onClick={() => { void viewer.current?.retry(); }}>Retry extra details</button> : null}
    {!state.error && !autoStart && state.enabled && state.status === "deferred" ? <button type="button" onClick={() => { void viewer.current?.start(); }}>Analyze extra details</button> : null}
  </aside>;
}
