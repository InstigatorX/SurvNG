import React, { useEffect, useRef, useState } from "react";
import { ChevronDown, CircleAlert, HardDrive, Radio, ShieldCheck } from "lucide-react";
import { formatBytes } from "../shared/format.js";
import { useRuntimeState } from "../shared/runtimeState.jsx";
import { formatDuration, recordingHealth } from "../recordingHealth.mjs";
import "./recordingHealth.css";

function ageLabel(value) {
  if (!value) return "Unavailable";
  const timestamp = typeof value === "number" && value < 10_000_000_000 ? value * 1000 : new Date(value).getTime();
  const age = Math.max(0, Date.now() - timestamp);
  if (!Number.isFinite(age)) return "Unavailable";
  return age < 60_000 ? "just now" : `${Math.floor(age / 60_000)}m ago`;
}

export function RecordingHealthBar() {
  const runtime = useRuntimeState();
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const buttonRef = useRef(null);
  const panelRef = useRef(null);
  const health = recordingHealth({ ...runtime, now });
  const storageLabel = health.storage.state === "unavailable" || !Number.isFinite(health.storage.free_bytes)
    ? "Storage unavailable" : `${formatBytes(health.storage.free_bytes)} free`;
  const recordingLabel = health.cameraDataKnown ? `${health.activeCount}/${health.expectedCount} recording` : runtime?.loading ? "Checking recording status" : "Recording status unavailable";
  const storageTitle = health.storage.state === "critical" ? "Storage critically low"
    : health.storage.state === "warning" ? "Storage below cleanup threshold" : "Storage";

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 10_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    function closeForOutside(event) {
      if (!panelRef.current?.contains(event.target) && !buttonRef.current?.contains(event.target)) setOpen(false);
    }
    function closeForEscape(event) {
      if (event.key === "Escape") { event.preventDefault(); setOpen(false); buttonRef.current?.focus(); }
    }
    document.addEventListener("pointerdown", closeForOutside);
    document.addEventListener("keydown", closeForEscape);
    return () => { document.removeEventListener("pointerdown", closeForOutside); document.removeEventListener("keydown", closeForEscape); };
  }, [open]);

  useEffect(() => {
    if (open) panelRef.current?.focus();
  }, [open]);

  return <div className="recording-health" aria-label="Recording health">
    <span className={`recording-health-item ${health.cameraDataKnown && health.activeCount === health.expectedCount ? "healthy" : "attention"}`} title={recordingLabel}><Radio size={15} /><strong>{recordingLabel}</strong></span>
    <span className={`recording-health-item ${health.storage.state}`} title={`${storageTitle}: ${storageLabel}`} aria-label={`${storageTitle}: ${storageLabel}`}><HardDrive size={15} /><strong>{storageLabel}</strong></span>
    <span className={`recording-health-item ${health.issues ? "attention" : "healthy"}`}><CircleAlert size={15} /><strong>{health.issues ? "Needs attention" : "No issues"}</strong></span>
    <button ref={buttonRef} type="button" className="recording-health-expand" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls="recording-health-details">
      Details <ChevronDown size={15} />
    </button>
    {open ? <section ref={panelRef} tabIndex={-1} id="recording-health-details" className="recording-health-popover" aria-label="Recording health details">
      <header><span><ShieldCheck size={16} /> Recording health</span><small>{health.cameraDataKnown && health.systemFresh ? "Current" : "Status may be stale"}</small></header>
      <div className="recording-health-summary"><span>Expected <strong>{health.cameraDataKnown ? health.expectedCount : "—"}</strong></span><span>Active <strong>{health.cameraDataKnown ? health.activeCount : "—"}</strong></span><span>Attention areas <strong>{health.issues}</strong></span></div>
      <div className="recording-health-storage"><strong>{storageTitle}</strong><span>{storageLabel}</span><small>{health.storage.sampled_at ? `Sampled ${ageLabel(health.storage.sampled_at)}` : "Storage sample unavailable"}{Number.isFinite(health.storage.freePercent) ? ` · ${health.storage.freePercent.toFixed(1)}% free` : ""}</small></div>
      <div className="recording-health-rows">{health.rows.length ? health.rows.map((row) => <div className={`recording-health-row ${row.state}`} key={row.id}><span><strong>{row.name}</strong><small>{row.state === "stale" ? "Last known status; refresh needed" : row.state === "healthy" ? "Recording" : row.state === "paused" ? "Recording paused" : row.state === "disabled" ? "Camera disabled" : row.missing.length ? `Missing ${row.missing.join(" and ")}` : "Runtime unavailable"}</small></span><em>{row.state === "stale" ? "last known" : row.state}</em></div>) : <div className="recording-health-empty">Camera status unavailable</div>}</div>
      <footer><span>System uptime: {formatDuration(health.uptimeSeconds)}</span><span>{health.cameraDataKnown ? "Camera status current" : "Camera status unavailable or stale"}</span></footer>
    </section> : null}
  </div>;
}
