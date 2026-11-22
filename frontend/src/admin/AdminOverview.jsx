import React, { useCallback, useEffect, useMemo, useState } from "react";
import { appUrl, fetch } from "../shared/api.js";
import { useRuntimeState } from "../shared/runtimeState.jsx";
import { recordingHealth } from "../recordingHealth.mjs";
import {
  formatOverviewBytes,
  overviewAttention,
  overviewCameraRows,
  pendingReviewSuggestions,
  recentChangeRows,
} from "../adminOverview.mjs";
import { Sparkline, UptimeStrip } from "./healthVisuals.jsx";
import "./overview.css";

const REFRESH_MS = 60_000;

async function optionalJson(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

export function AdminOverview({ config, detectorStatus = null, retentionStatus = null, onAction }) {
  const runtime = useRuntimeState();
  const [telemetry, setTelemetry] = useState(null);
  const [availability, setAvailability] = useState(null);
  const [review, setReview] = useState(null);
  const [retention, setRetention] = useState(null);
  // undefined while loading; null when the history could not be read.
  const [changes, setChanges] = useState(undefined);
  const [now, setNow] = useState(() => Date.now());

  const refresh = useCallback(async () => {
    const [telemetryPayload, availabilityPayload, reviewPayload, retentionPayload, changesPayload] = await Promise.all([
      optionalJson("/api/telemetry?hours=24"),
      optionalJson("/api/telemetry/availability?hours=24"),
      optionalJson("/api/system-review"),
      optionalJson("/api/retention/status"),
      optionalJson("/api/config/changes?limit=5"),
    ]);
    setChanges(changesPayload);
    setRetention(retentionPayload);
    setTelemetry(telemetryPayload);
    setAvailability(availabilityPayload);
    setReview(reviewPayload);
    setNow(Date.now());
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const health = useMemo(() => recordingHealth({ ...runtime, now }), [runtime, now]);
  const detector = runtime.system?.detector || detectorStatus || null;
  const attention = useMemo(() => overviewAttention({
    health,
    detector,
    retention: retention || retentionStatus,
    reviewSuggestions: pendingReviewSuggestions(review),
    interruptions: telemetry?.interruption_summary,
  }), [health, detector, retention, retentionStatus, review, telemetry]);
  const cameras = config?.cameras || [];
  const rows = useMemo(
    () => overviewCameraRows({ health, cameras, availability: availability?.cameras }),
    [health, cameras, availability],
  );

  const recentRows = useMemo(() => recentChangeRows(changes?.changes, now), [changes, now]);
  const expected = health.expectedCount;
  const active = health.activeCount;
  const offlineRows = health.rows.filter((row) => row.state === "attention" || row.state === "unavailable");
  const storage = telemetry?.system?.storage || null;
  const usedBytes = Number(storage?.used_bytes);
  const totalBytes = Number(storage?.total_bytes);
  const usedPercent = totalBytes > 0 && Number.isFinite(usedBytes) ? (usedBytes / totalBytes) * 100 : null;
  const storageTone = health.storage.state === "critical" ? "bad" : health.storage.state === "warning" ? "warn" : "good";
  const inferenceMs = Number(detector?.runtime?.average_inference_ms);
  const backend = [detector?.loaded_backend, detector?.loaded_device].filter(Boolean).join(" · ");
  const hourly = Array.isArray(telemetry?.activity?.hourly) ? telemetry.activity.hourly : [];
  const incidents24h = telemetry?.activity?.last_24h?.object_incidents;
  const summary = !health.cameraDataKnown
    ? "Camera status is unavailable."
    : `${active} of ${expected} camera${expected === 1 ? "" : "s"} recording. ${attention.length ? `${attention.length} item${attention.length === 1 ? " needs" : "s need"} a decision.` : "Nothing needs attention."}`;

  return (
    <section id="admin-panel-home" className="admin-overview" aria-labelledby="admin-home-title">
      <header className="admin-overview-head">
        <div>
          <h2 id="admin-home-title">Overview</h2>
          <p>{summary}</p>
        </div>
        <div className="admin-overview-head-actions">
          <button type="button" onClick={() => onAction?.({ id: "add-camera" })}>Add camera</button>
          <a className="button" href={appUrl("/api/support-bundle")} download="survng-support-bundle.json">Download support bundle</a>
        </div>
      </header>

      <section className="admin-overview-tiles" aria-label="System summary">
        <article>
          <span className="label">Cameras recording</span>
          <strong>{health.cameraDataKnown ? <>{active}<small>/{expected}</small></> : "—"}</strong>
          {offlineRows.length
            ? <span className="admin-overview-chip bad"><i />{offlineRows.length === 1 ? `${offlineRows[0].name} not recording` : `${offlineRows.length} not recording`}</span>
            : health.cameraDataKnown ? <span className="admin-overview-chip good"><i />All recording</span> : null}
        </article>
        <article>
          <span className="label">Detector</span>
          <strong>{detector?.enabled === false ? "Off" : Number.isFinite(inferenceMs) && inferenceMs > 0 ? <>{Math.round(inferenceMs)}<small> ms avg</small></> : "—"}</strong>
          {detector?.enabled === false
            ? <span className="admin-overview-chip muted">Disabled</span>
            : backend ? <span className="admin-overview-chip good mono">{backend}</span>
              : <span className="admin-overview-chip bad"><i />Not loaded</span>}
        </article>
        <article>
          <span className="label">Recordings volume</span>
          <strong>{Number.isFinite(usedBytes) && totalBytes > 0 ? <>{formatOverviewBytes(usedBytes).replace(/ \w+$/, "")}<small> / {formatOverviewBytes(totalBytes)}</small></> : "—"}</strong>
          {usedPercent !== null
            ? <div className={`admin-overview-meter ${storageTone}`} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(usedPercent)} aria-label="Storage used"><span style={{ width: `${Math.min(100, usedPercent)}%` }} /></div>
            : null}
        </article>
        <article>
          <span className="label">Incidents · 24 h</span>
          <strong>{Number.isFinite(Number(incidents24h)) ? Number(incidents24h).toLocaleString() : "—"}</strong>
          <Sparkline values={hourly.map((item) => Number(item.object_incidents || 0))} label="Object incidents per hour, last 24 hours" />
        </article>
      </section>

      <div className="admin-overview-split">
        <section className="admin-overview-panel" aria-labelledby="admin-overview-attention">
          <div className="admin-overview-panel-head">
            <h2 id="admin-overview-attention">Needs attention</h2>
            <span>Sorted by impact on recording</span>
          </div>
          {attention.length ? attention.map((item) => (
            <div className={`admin-overview-attention ${item.severity}`} key={item.id}>
              <div>
                <strong>{item.title}</strong>
                {item.detail ? <p>{item.detail}</p> : null}
              </div>
              <div className="admin-overview-actions">
                {item.actions.map((action) => <button type="button" key={action.id} className={action.primary ? "primary" : undefined} onClick={() => onAction?.(action)}>{action.label}</button>)}
              </div>
            </div>
          )) : <p className="admin-overview-empty">Nothing needs attention. Cameras are recording, storage is above the cleanup threshold and detection is running.</p>}
        </section>
        <section className="admin-overview-panel" aria-labelledby="admin-overview-recent">
          <div className="admin-overview-panel-head">
            <h2 id="admin-overview-recent">Recently changed</h2>
          </div>
          {changes === undefined
            ? <p className="admin-overview-empty">Loading recent changes…</p>
            : changes === null
            ? <p className="admin-overview-empty">Change history is unavailable.</p>
            : recentRows.length ? <ul className="admin-overview-changes">
              {recentRows.map((row) => (
                <li key={row.id}>
                  <span className="admin-overview-change-label">{row.label}</span>
                  <span className="mono muted">{row.values}{row.more ? <em> {row.more}</em> : null}</span>
                  <span className="admin-overview-change-meta">{row.meta}</span>
                </li>
              ))}
            </ul> : <p className="admin-overview-empty">No changes saved yet. Settings you save from Admin appear here.</p>}
        </section>
      </div>

      <section className="admin-overview-panel" aria-labelledby="admin-overview-cameras">
        <div className="admin-overview-panel-head">
          <h2 id="admin-overview-cameras">Cameras</h2>
          <span>Availability, last 24 h</span>
        </div>
        {rows.length ? (
          <div className="admin-overview-table-scroll">
            <table className="admin-overview-table">
              <thead><tr><th scope="col">Camera</th><th scope="col">Status</th><th scope="col">Main stream</th><th scope="col">Detection</th><th scope="col">HomeKit</th><th scope="col" className="uptime-col">24 h</th></tr></thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <th scope="row"><button type="button" className="admin-overview-link" onClick={() => onAction?.({ id: "open-camera", cameraId: row.id })}>{row.name}</button></th>
                    <td><span className={`admin-overview-chip ${row.status.tone}`}><i />{row.status.label}</span></td>
                    <td className="mono muted">{row.stream}</td>
                    <td>{row.detection}</td>
                    <td className={row.homekit === "Off" ? "muted" : undefined}>{row.homekit}</td>
                    <td><UptimeStrip slots={row.uptime} name={row.name} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="admin-overview-empty">No cameras yet. Add a camera to start recording.</p>}
      </section>
    </section>
  );
}
