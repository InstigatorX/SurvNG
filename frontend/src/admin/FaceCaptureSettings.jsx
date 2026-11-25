import React, { useEffect, useState } from "react";
import { CircleAlert, History, Pause, Play, RefreshCcw, ScanFace, X } from "lucide-react";
import { fetch } from "../shared/api.js";

const STATUS_REFRESH_MS = 5000;
const DAY_MS = 24 * 60 * 60 * 1000;

function apiDetail(payload, fallback) {
  const detail = payload?.detail;
  return typeof detail === "string" && detail ? detail : fallback;
}

// datetime-local works in the browser's local time without a zone.
function localInputValue(date) {
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function isoFromLocal(value) {
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : "";
}

function when(epoch) {
  return Number.isFinite(epoch) ? new Date(epoch * 1000).toLocaleString() : "";
}

function cameraSummary(seed, cameras) {
  if (!seed.cameras?.length) return "All cameras";
  const names = new Map((cameras || []).map((camera) => [camera.id, camera.name]));
  return seed.cameras.map((id) => names.get(id) || id).join(", ");
}

function useFaceCaptureStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  async function load() {
    try {
      const response = await fetch("/api/face-capture/status");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Face capture status is unavailable."));
      setStatus(payload);
      setError("");
    } catch (reason) {
      setError(reason.message || "Face capture status is unavailable.");
    }
  }
  useEffect(() => {
    void load();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, STATUS_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, []);
  return { status, error, reload: load };
}

function SeedProgress({ seed, cameras, busy, onAction }) {
  const total = Math.max(seed.total || 0, seed.processed || 0);
  const percent = total ? Math.round((seed.processed / total) * 100) : 0;
  return <div className="face-capture-seed-active" role="status">
    <div className="face-capture-seed-head">
      <strong>{seed.state === "paused" ? "Paused" : "Processing previous incidents"}</strong>
      <span>{when(seed.start_epoch)} → {when(seed.end_epoch)} · {cameraSummary(seed, cameras)}</span>
    </div>
    <progress max={total || 1} value={seed.processed || 0} aria-label="Seed progress" />
    <div className="face-capture-seed-numbers">
      <span>{(seed.processed || 0).toLocaleString()} of {total.toLocaleString()} events ({percent}%)</span>
      <span>{(seed.faces || 0).toLocaleString()} faces kept</span>
      {seed.failed ? <span>{seed.failed.toLocaleString()} could not be read</span> : null}
    </div>
    <div className="face-capture-seed-actions">
      {seed.state === "running"
        ? <button type="button" disabled={busy} onClick={() => onAction(seed.id, "pause")}><Pause size={14} />Pause</button>
        : <button type="button" disabled={busy} onClick={() => onAction(seed.id, "resume")}><Play size={14} />Resume</button>}
      <button type="button" className="danger" disabled={busy} onClick={() => onAction(seed.id, "cancel")}><X size={14} />Cancel</button>
    </div>
    <small className="settings-help">New events are always captured first; this runs one past event at a time, behind incident work.</small>
  </div>;
}

function SeedForm({ status, busy, onStart }) {
  const now = new Date();
  const [start, setStart] = useState(() => localInputValue(new Date(now.getTime() - 14 * DAY_MS)));
  const [end, setEnd] = useState(() => localInputValue(now));
  const [allCameras, setAllCameras] = useState(true);
  const [selected, setSelected] = useState(() => new Set());
  const cameras = status?.cameras || [];
  const startIso = isoFromLocal(start);
  const endIso = isoFromLocal(end);
  const rangeValid = Boolean(startIso && endIso && startIso < endIso);
  const camerasValid = allCameras || selected.size > 0;
  const enabled = Boolean(status?.enabled);

  function toggle(id) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  return <form className="face-capture-seed-form" onSubmit={(event) => {
    event.preventDefault();
    if (rangeValid && camerasValid) onStart({ start: startIso, end: endIso, cameras: allCameras ? [] : [...selected] });
  }}>
    <div className="face-capture-seed-range">
      <label>From<input type="datetime-local" value={start} max={end} onChange={(event) => setStart(event.target.value)} required /></label>
      <label>To<input type="datetime-local" value={end} min={start} onChange={(event) => setEnd(event.target.value)} required /></label>
    </div>
    <fieldset className="face-capture-cameras">
      <legend>Cameras</legend>
      <label className="compact-toggle"><input type="radio" name="face-capture-cameras" checked={allCameras} onChange={() => setAllCameras(true)} /><span>All cameras</span></label>
      <label className="compact-toggle"><input type="radio" name="face-capture-cameras" checked={!allCameras} onChange={() => setAllCameras(false)} /><span>Choose cameras</span></label>
      {!allCameras ? <div className="face-capture-camera-list">
        {cameras.map((camera) => <label className="compact-toggle" key={camera.id}>
          <input type="checkbox" checked={selected.has(camera.id)} onChange={() => toggle(camera.id)} /><span>{camera.name}</span>
        </label>)}
      </div> : null}
    </fieldset>
    {!rangeValid ? <p className="settings-help" role="alert">The start must be before the end.</p> : null}
    {!enabled ? <p className="settings-help"><CircleAlert size={14} /> Turn on face capture and save before processing previous incidents.</p> : null}
    <button type="submit" className="primary" disabled={busy || !enabled || !rangeValid || !camerasValid}><Play size={14} />{busy ? "Counting events…" : "Start"}</button>
  </form>;
}

function SeedHistory({ seeds, cameras }) {
  const finished = (seeds || []).filter((seed) => seed.state === "done" || seed.state === "cancelled");
  if (!finished.length) return null;
  return <details className="face-capture-seed-history">
    <summary><History size={14} />Earlier runs</summary>
    <ul>
      {finished.map((seed) => <li key={seed.id}>
        <strong>{seed.state === "done" ? "Finished" : "Cancelled"}</strong>
        <span>{when(seed.start_epoch)} → {when(seed.end_epoch)} · {cameraSummary(seed, cameras)}</span>
        <span>{(seed.processed || 0).toLocaleString()} events · {(seed.faces || 0).toLocaleString()} faces{seed.failed ? ` · ${seed.failed} unreadable` : ""}</span>
      </li>)}
    </ul>
  </details>;
}

export function FaceCaptureSettings({ config, updateConfig }) {
  const { status, error, reload } = useFaceCaptureStatus();
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const detector = config.detector || {};
  const activeSeed = (status?.seeds || []).find((seed) => seed.state === "running" || seed.state === "paused");

  async function post(path, body) {
    setBusy(true);
    setActionError("");
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "The request failed."));
      await reload();
    } catch (reason) {
      setActionError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  const number = (key, fallback) => detector[key] ?? fallback;
  return <details className="detection-settings-card detection-feature-card wide-card face-capture-card" open>
    <summary><span className="detection-settings-card-icon"><ScanFace size={18} /></span><span><strong>Face capture</strong><small>Keep each person's best faces from recorded person events, in the background.</small></span></summary>
    <div className="detection-feature-body detection-field-grid">
      <label className="compact-toggle"><input type="checkbox" checked={detector.face_capture_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_capture_enabled"], event.target.checked)} /><span>Face capture enabled</span></label>
      <label>Minimum face quality<input type="number" min="0" max="40" step="0.5" value={number("face_capture_min_norm", 13)} onChange={(event) => updateConfig(["detector", "face_capture_min_norm"], Number(event.target.value))} /><small>AdaFace quality score; real faces measured 13–19 here, backs of heads and leaves 9–12. Default 13.</small></label>
      <label>Minimum face size (px)<input type="number" min="16" max="1024" step="1" value={number("face_capture_min_size", 48)} onChange={(event) => updateConfig(["detector", "face_capture_min_size"], Number(event.target.value))} /></label>
      <label>Faces kept per person<input type="number" min="1" max="16" step="1" value={number("face_capture_faces_per_track", 3)} onChange={(event) => updateConfig(["detector", "face_capture_faces_per_track"], Number(event.target.value))} /></label>
      <label>Catch up on the last (hours)<input type="number" min="1" max="720" step="1" value={number("face_capture_backfill_hours", 24)} onChange={(event) => updateConfig(["detector", "face_capture_backfill_hours"], Number(event.target.value))} /><small>New events within this window are captured automatically. Use “Process previous incidents” for older ones.</small></label>
    </div>
    <div className="face-capture-status">
      {error ? <p className="settings-help" role="alert"><CircleAlert size={14} /> {error} <button type="button" className="link-button" onClick={() => void reload()}><RefreshCcw size={13} />Retry</button></p> : null}
      {status ? <p className="settings-help">
        {status.enabled ? (status.running ? "Running" : "Starting") : "Off"}
        {" · "}{(status.faces || 0).toLocaleString()} faces stored
        {status.events?.done ? ` · ${status.events.done.toLocaleString()} events captured` : ""}
        {status.events?.failed ? ` · ${status.events.failed.toLocaleString()} unreadable` : ""}
        {status.last_error ? <><br /><CircleAlert size={13} /> {status.last_error}</> : null}
      </p> : !error ? <p className="settings-help">Loading face capture status…</p> : null}
    </div>
    <section className="face-capture-seed">
      <h4>Process previous incidents</h4>
      {actionError ? <p className="settings-help" role="alert"><CircleAlert size={14} /> {actionError}</p> : null}
      {activeSeed
        ? <SeedProgress seed={activeSeed} cameras={status?.cameras} busy={busy} onAction={(id, action) => void post(`/api/face-capture/seeds/${id}/${action}`)} />
        : <SeedForm status={status} busy={busy} onStart={(body) => void post("/api/face-capture/seeds", body)} />}
      <SeedHistory seeds={status?.seeds} cameras={status?.cameras} />
    </section>
  </details>;
}
