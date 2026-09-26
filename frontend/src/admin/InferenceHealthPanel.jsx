import { inferenceTargetRows, formatAttemptOutcome, formatInferenceMs, formatRoleAttempts } from "./inferenceWorkers.mjs";

function stateLabel(row) {
  if (row.kind === "primary") return row.ready ? "Loaded" : "Not loaded";
  return row.ready ? "Ready" : "Connected";
}

export function InferenceHealthPanel({ detector, detectorConfig }) {
  if (!detector) return <div className="empty-state">Waiting for telemetry...</div>;
  const registry = detector?.remote_registry || {};
  const rows = inferenceTargetRows(detector, detectorConfig);
  const mode = detector?.inference_mode || detectorConfig?.inference_mode || "local";
  const balance = detectorConfig?.inference_balance || "remote_first";
  const balanceLabel = balance === "weighted" ? "Weighted share" : "Remote first";
  return (
    <section className="telemetry-section inference-health-section">
      <div className="telemetry-section-head">
        <div>
          <h3>Inference workers</h3>
          <p>Live load and latency for this server and every connected inference worker. Weights are changed under Object Detection → Workers.</p>
        </div>
      </div>
      <div className="telemetry-summary-grid overview-summary">
        <article><span>Execution</span><strong>{mode}</strong><small>{balanceLabel}</small></article>
        <article><span>Workers</span><strong>{Number(registry.ready || 0)}/{Number(registry.connected || 0)}</strong><small>{registry.accepting === false ? "Not accepting" : "Accepting connections"}</small></article>
        <article><span>Primary device</span><strong>{detector?.loaded_device || detector?.configured_device || "—"}</strong><small>{formatInferenceMs(detector?.runtime?.average_inference_ms)} average</small></article>
        <article><span>Primary queue</span><strong>{Number(detector?.runtime?.queue_depth || 0)}</strong><small>{Number(detector?.runtime?.failed_inferences || 0)} failed inferences</small></article>
      </div>
      <div className="inference-target-list">
        {rows.map((row) => (
          <article className="inference-target-card" key={row.id}>
            <header>
              <strong>{row.name}</strong>
              <em className={row.ready ? "good" : "attention"}>{stateLabel(row)}</em>
            </header>
            <dl className="telemetry-details">
              <div><dt>Device</dt><dd>{row.device}</dd></div>
              <div><dt>Roles</dt><dd>{row.roles.length ? row.roles.join(", ") : "—"}</dd></div>
              <div><dt>Weight</dt><dd>{row.weight}</dd></div>
              <div><dt>In progress</dt><dd>{row.pending}</dd></div>
              <div><dt>{row.kind === "primary" ? "Inferences" : "Completed requests"}</dt><dd>{row.completed.toLocaleString()}</dd></div>
              {row.kind === "worker" ? <div><dt>Rerouted</dt><dd>{row.rerouted.toLocaleString()}</dd></div> : null}
              <div><dt>Failed</dt><dd className={row.failed ? "attention" : undefined}>{row.failed.toLocaleString()}</dd></div>
              <div><dt>Last inference</dt><dd>{formatInferenceMs(row.lastInferenceMs)}</dd></div>
              <div><dt>Average inference</dt><dd>{formatInferenceMs(row.averageInferenceMs)}</dd></div>
              {row.kind === "worker" ? <div><dt>Round trip</dt><dd>{formatInferenceMs(row.lastRequestMs)}</dd></div> : null}
              <div><dt>Model load</dt><dd>{formatInferenceMs(row.modelLoadMs)}</dd></div>
              <div><dt>Lease</dt><dd>{row.kind === "primary" ? "Local" : (Number.isFinite(row.leaseSeconds) ? `${row.leaseSeconds.toFixed(0)}s` : "—")}</dd></div>
            </dl>
            {row.routingHold ? <p className="inference-target-note attention">Paused until its queue finishes.</p> : null}
            {formatRoleAttempts(row.roleAttempts) ? <p className="inference-target-note">{formatRoleAttempts(row.roleAttempts)}</p> : null}
            {formatAttemptOutcome(row) ? <p className={row.lastOutcome === "failed" ? "inference-target-note attention" : "inference-target-note"}>{formatAttemptOutcome(row)}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}
