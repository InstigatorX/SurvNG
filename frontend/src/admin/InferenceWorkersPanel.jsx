import { useEffect, useState } from "react";
import { fetch } from "../shared/api.js";
import { inferenceTargetRows, formatAttemptOutcome, formatInferenceMs, formatRoleAttempts } from "./inferenceWorkers.mjs";

export function InferenceWorkersPanel({ config, updateConfig, detectorStatus }) {
  const [liveStatus, setLiveStatus] = useState(detectorStatus || null);
  const detector = config?.detector || {};
  const mode = detector.inference_mode || "local";
  const balance = detector.inference_balance || "remote_first";
  const weighted = balance === "weighted" && mode !== "local";
  const status = liveStatus || detectorStatus;
  const rows = inferenceTargetRows(status, detector).filter((row) => row.kind === "worker");

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const response = await fetch("/api/detector/status", { cache: "no-store" });
        if (!response.ok || cancelled) return;
        const payload = await response.json();
        if (!cancelled) setLiveStatus(payload);
      } catch {
        // Keep the last status. The panel already explains an empty worker list.
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  function setWeight(workerId, value) {
    const next = { ...(detector.inference_worker_weights || {}) };
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    next[workerId] = Math.max(0, Math.min(100, Math.round(parsed)));
    updateConfig(["detector", "inference_worker_weights"], next);
  }

  return (
    <section className="detection-settings-card wide-card">
      <header className="detection-settings-card-head">
        <div>
          <h3>Inference workers</h3>
          <p>Share detections between this server and connected workers, or keep workers in front and use this server only as fallback. Rerouted requests finished on another target. Failed counts requests that no target completed.</p>
        </div>
      </header>
      <div className="detection-field-grid">
        <label>Load balancing
          <select value={balance} onChange={(event) => updateConfig(["detector", "inference_balance"], event.target.value)} disabled={mode === "local"}>
            <option value="remote_first">Remote first</option>
            <option value="weighted">Weighted share</option>
          </select>
          <small>{mode === "local" ? "Switch inference execution to hybrid or remote before balancing." : "Remote first sends work to workers. Weighted share uses the weights below and the current queue."}</small>
        </label>
        <label>Primary weight
          <input type="number" min="0" max="100" step="1" value={detector.inference_primary_weight ?? 1} disabled={!weighted || mode === "remote"} onChange={(event) => updateConfig(["detector", "inference_primary_weight"], Number(event.target.value))} />
          <small>{mode === "remote" ? "Remote execution does not run detections on this server." : "0 skips this server. Equal weights share evenly when both queues are idle."}</small>
        </label>
        <label>Default worker weight
          <input type="number" min="0" max="100" step="1" value={detector.inference_default_worker_weight ?? 1} disabled={!weighted} onChange={(event) => updateConfig(["detector", "inference_default_worker_weight"], Number(event.target.value))} />
          <small>Used for a connected worker until it has its own weight. 0 disables that worker.</small>
        </label>
      </div>
      <div className="inference-target-list">
        {rows.length ? rows.map((row) => (
          <article className="inference-target-card" key={row.id}>
            <header>
              <strong>{row.name}</strong>
              <em className={row.ready ? "good" : "attention"}>{row.ready ? "Ready" : "Not ready"}</em>
            </header>
            <dl className="telemetry-details">
              <div><dt>Device</dt><dd>{row.device}</dd></div>
              <div><dt>Roles</dt><dd>{row.roles.join(", ") || "—"}</dd></div>
              <div><dt>In progress</dt><dd>{row.pending}</dd></div>
              <div><dt>Completed</dt><dd>{row.completed.toLocaleString()}</dd></div>
              <div><dt>Rerouted</dt><dd>{row.rerouted.toLocaleString()}</dd></div>
              <div><dt>Failed</dt><dd className={row.failed ? "attention" : undefined}>{row.failed.toLocaleString()}</dd></div>
              <div><dt>Last inference</dt><dd>{formatInferenceMs(row.lastInferenceMs)}</dd></div>
              <div><dt>Round trip</dt><dd>{formatInferenceMs(row.lastRequestMs)}</dd></div>
              <div><dt>Lease</dt><dd>{Number.isFinite(row.leaseSeconds) ? `${row.leaseSeconds.toFixed(0)}s` : "—"}</dd></div>
            </dl>
            {formatRoleAttempts(row.roleAttempts) ? <p className="inference-target-note">{formatRoleAttempts(row.roleAttempts)}</p> : null}
            {formatAttemptOutcome(row) ? <p className={row.lastOutcome === "failed" ? "inference-target-note attention" : "inference-target-note"}>{formatAttemptOutcome(row)}</p> : null}
            <label>Weight
              <input type="number" min="0" max="100" step="1" value={row.weight} disabled={!weighted} onChange={(event) => setWeight(row.id, event.target.value)} />
            </label>
          </article>
        )) : <p className="settings-help">No inference workers are connected. A worker appears here after it authenticates and finishes loading models.</p>}
      </div>
    </section>
  );
}
