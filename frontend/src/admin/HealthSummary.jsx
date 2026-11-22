import React, { useEffect, useMemo, useState } from "react";
import { fetch } from "../shared/api.js";
import { formatBytes } from "../shared/format.js";
import { healthCameraRows, healthFindings, latestValue, memoryTrendBytesPerHour, seriesValues, trackingCapacitySummary } from "../healthSummary.mjs";
import { Sparkline, UptimeStrip } from "./healthVisuals.jsx";
import "./overview.css";

const RANGES = [
  { id: "short", label: "2 h" },
  { id: "long", label: "7 d" },
];

function percent(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(value === 100 ? 0 : digits)}%` : "—";
}

/**
 * Health at a glance: tiles, findings, camera reliability and pipeline trends.
 * The detailed telemetry stays available below it.
 */
export function HealthSummary({ data, cameraId = "", onAction }) {
  const [availability, setAvailability] = useState(null);
  const [range, setRange] = useState("short");
  const generatedAt = data?.generated_at || "";

  useEffect(() => {
    let cancelled = false;
    fetch("/api/telemetry/availability?hours=24")
      .then((response) => (response.ok ? response.json() : null))
      .catch(() => null)
      .then((payload) => { if (!cancelled) setAvailability(payload); });
    return () => { cancelled = true; };
  }, [generatedAt]);

  const rows = useMemo(() => healthCameraRows(data, availability?.cameras, cameraId), [data, availability, cameraId]);
  const findings = useMemo(() => healthFindings({ data: cameraId ? { ...data, interruption_summary: null, detector: null, tracking_capacity_history: null } : data, rows }), [data, rows, cameraId]);
  if (!data) return null;

  const history = data.runtime_history?.[range] || [];
  const tracking = data.tracking_capacity_history?.[range] || [];
  const memoryHistory = data.process_memory_history?.long || [];
  const known = rows.map((row) => row.uptimePercent).filter((value) => value !== null);
  const reliability = known.length ? known.reduce((total, value) => total + value, 0) / known.length : null;
  const worst = rows.filter((row) => row.uptimePercent !== null).sort((left, right) => left.uptimePercent - right.uptimePercent)[0];
  const cpu = latestValue(data.runtime_history?.short, "cpu_load_percent");
  const applicationBytes = Number(data.system?.service_memory?.application_bytes);
  const growth = memoryTrendBytesPerHour(memoryHistory);
  const growthLabel = growth === null ? null : `${growth >= 0 ? "+" : "−"}${formatBytes(Math.abs(growth))}/h`;
  const rising = growth !== null && growth > 50 * 1024 ** 2;
  const activity = (cameraId ? data.cameras?.find((camera) => camera.id === cameraId)?.activity : data.activity) || {};
  const lastDay = activity.last_24h || {};
  const events = Number(lastDay.events || 0);
  const incidents = Number(lastDay.object_incidents || 0);
  const runtime = data.detector?.runtime || {};
  const semantic = data.semantic_search || {};
  const coverage = latestValue(history, "analysis_coverage_percent");
  const inferenceMs = Number(runtime.average_inference_ms) || 0;
  const detectorLabel = [data.detector?.loaded_backend, data.detector?.loaded_device].filter(Boolean).join(" on ") || "Object detector";
  const semanticState = String(semantic.state || (semantic.enabled ? "starting" : "disabled"));
  const trackingCapacity = trackingCapacitySummary(tracking);
  const onTimePercent = trackingCapacity.onTimePercent;
  const rangeLabel = range === "short" ? "last 2 hours" : "last 7 days";
  const offline = rows.filter((row) => row.offlineNow);
  const summary = !rows.length
    ? "No camera telemetry yet."
    : `Last 24 hours across ${rows.length} camera${rows.length === 1 ? "" : "s"}. ${offline.length ? `${offline.length} camera${offline.length === 1 ? " is" : "s are"} offline now${findings.length > offline.length ? ", and other items need a look" : ""}.` : findings.length ? `${findings.length} item${findings.length === 1 ? " is" : "s are"} outside the normal range.` : "Everything is within the normal range."}`;

  return (
    <section className="admin-overview health-summary" aria-labelledby="health-summary-title">
      <header className="admin-overview-head">
        <div>
          <h2 id="health-summary-title">Health</h2>
          <p>{summary}</p>
        </div>
        <div className="health-range" role="group" aria-label="Trend range">
          {RANGES.map((item) => <button type="button" key={item.id} aria-pressed={range === item.id} className={range === item.id ? "active" : undefined} onClick={() => setRange(item.id)}>{item.label}</button>)}
        </div>
      </header>

      <section className="admin-overview-tiles" aria-label="Health summary">
        <article>
          <span className="label">Camera reliability · 24 h</span>
          <strong>{reliability === null ? "—" : <>{reliability.toFixed(1)}<small>%</small></>}</strong>
          {worst && worst.uptimePercent < 99
            ? <span className={`admin-overview-chip ${worst.uptimePercent < 90 ? "bad" : "warn"}`}><i />{worst.name} {percent(worst.uptimePercent)}</span>
            : reliability !== null ? <span className="admin-overview-chip good"><i />All cameras ≥ 99%</span> : null}
        </article>
        <article>
          <span className="label">System load</span>
          <strong>{cpu === null ? "—" : <>{Math.round(cpu)}<small>% CPU</small></>}</strong>
          <Sparkline values={seriesValues(history, "cpu_load_percent")} label={`CPU load, ${rangeLabel}`} emptyText="No samples yet" />
        </article>
        <article>
          <span className="label">Application memory</span>
          <strong>{Number.isFinite(applicationBytes) && applicationBytes > 0 ? formatBytes(applicationBytes) : "—"}</strong>
          {growthLabel ? <span className={`admin-overview-chip ${rising ? "warn" : "good"}`}><i />{rising ? "Rising" : "Stable"} · {growthLabel}</span> : null}
        </article>
        <article>
          <span className="label">Events · 24 h</span>
          <strong>{incidents.toLocaleString()}<small> incidents</small></strong>
          <span className="admin-overview-muted">from {events.toLocaleString()} motion events{events ? ` · ${Math.round(((events - Math.min(incidents, events)) / events) * 100)}% filtered` : ""}</span>
        </article>
        {cameraId ? null : <>
          <article title={detectorLabel}>
            <span className="label">Detector response</span>
            <strong>{inferenceMs > 0 ? <>{Math.round(inferenceMs)}<small> ms avg</small></> : "—"}</strong>
            <Sparkline values={seriesValues(history, "inference_ms")} label={`Detector response time, ${rangeLabel}`} emptyText="No samples yet" />
          </article>
          <article>
            <span className="label">Motion analysis</span>
            <strong>{coverage === null ? "Off" : <>{Math.round(coverage)}<small>% analyzed</small></>}</strong>
            <Sparkline values={seriesValues(history, "analysis_coverage_percent")} label={`Analysis coverage, ${rangeLabel}`} emptyText="Not active" />
          </article>
          <article title={trackingCapacity.requests ? `${trackingCapacity.requests.toLocaleString()} tracking requests · ${trackingCapacity.waited.toLocaleString()} waited for a free slot${trackingCapacity.waited ? ` (longest ${trackingCapacity.longestWaitSeconds.toFixed(1)} s)` : ""} · ${trackingCapacity.skipped.toLocaleString()} skipped` : undefined}>
            <span className="label">Object tracking</span>
            <strong>{onTimePercent === null ? "—" : <>{onTimePercent >= 99.95 ? "100" : onTimePercent.toFixed(1).replace(/\.0$/, "")}<small>% on time</small></>}</strong>
            {trackingCapacity.skipped ? <span className="admin-overview-chip bad"><i />{trackingCapacity.skipped.toLocaleString()} skipped</span> : null}
            <Sparkline values={trackingCapacity.lateSeries} label={`Tracking requests that waited or were skipped, ${rangeLabel}`} emptyText="No requests yet" />
          </article>
          <article>
            <span className="label">Semantic search index</span>
            <strong>{Number(semantic.queue_depth || 0).toLocaleString()}<small> queued</small></strong>
            <span className={`admin-overview-chip ${semanticState === "ready" ? "good" : semanticState === "disabled" ? "muted" : "warn"}`}><i />{semanticState.replaceAll("_", " ")}</span>
          </article>
        </>}
      </section>

      <section className="admin-overview-panel" aria-labelledby="health-findings">
        <div className="admin-overview-panel-head">
          <h2 id="health-findings">Findings</h2>
          <span>Only items outside the normal range</span>
        </div>
        {findings.length ? findings.map((item) => (
          <div className={`admin-overview-attention ${item.severity}`} key={item.id}>
            <div><strong>{item.title}</strong><p>{item.detail}</p></div>
            <div className="admin-overview-actions">
              {item.actions.map((action) => <button type="button" key={action.id} onClick={() => onAction?.(action)}>{action.label}</button>)}
            </div>
          </div>
        )) : <p className="admin-overview-empty">No findings. Cameras, detection and the service are within their normal range.</p>}
      </section>

      <section className="admin-overview-panel" aria-labelledby="health-reliability">
        <div className="admin-overview-panel-head">
          <h2 id="health-reliability">Camera reliability</h2>
          <span>Availability by hour</span>
        </div>
        {rows.length ? (
          <div className="admin-overview-table-scroll">
            <table className="admin-overview-table">
              <thead><tr><th scope="col">Camera</th><th scope="col">Uptime · 24 h</th><th scope="col">Reconnects</th><th scope="col">Interruptions · 24 h</th><th scope="col">Events · 24 h</th><th scope="col" className="uptime-col">24 h</th></tr></thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <th scope="row"><button type="button" className="admin-overview-link" onClick={() => onAction?.({ id: "open-camera", cameraId: row.id })}>{row.name}</button></th>
                    <td className="mono">{percent(row.uptimePercent)}</td>
                    <td className="mono">{row.reconnects.toLocaleString()}</td>
                    <td className="mono">{row.interruptions.toLocaleString()}</td>
                    <td className="mono">{row.events.toLocaleString()}</td>
                    <td><UptimeStrip slots={row.uptime} name={row.name} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="admin-overview-empty">No camera telemetry yet.</p>}
      </section>

    </section>
  );
}
