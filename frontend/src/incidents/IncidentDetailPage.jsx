import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Camera, ImageOff, Play, RefreshCw, X } from "lucide-react";
import { incidentEvidenceFrames } from "../incidentNavigation.mjs";
import { appUrl, fetch } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { IncidentClipLayer } from "./IncidentCard.jsx";
import "./incident-detail.css";

function EvidenceImage({ eventId, revision, label, onClick, src }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [eventId, revision, src]);
  const content = (!eventId && !src) || failed
    ? <span className="incident-detail-image-empty"><ImageOff size={28} />Image unavailable</span>
    : <img loading={onClick ? "lazy" : "eager"} src={appUrl(src || `/api/events/${eventId}/thumbnail.jpg?width=1280&quality=85&revision=${encodeURIComponent(revision || 0)}`)} alt={label} onError={() => setFailed(true)} />;
  return onClick
    ? <button className="incident-detail-frame" onClick={onClick} aria-label={label}>{content}<span>{label}</span></button>
    : <div className="incident-detail-hero">{content}</div>;
}

function EvidenceViewer({ frames, selected, revision, onClose }) {
  const dialog = useRef(null);
  const [index, setIndex] = useState(selected);
  useEffect(() => {
    dialog.current.showModal();
  }, []);
  const frame = frames[index];
  return <dialog ref={dialog} className="incident-evidence-viewer" aria-label="Evidence image viewer" onCancel={onClose} onClose={onClose}>
    <header><span>{frame.label} · {index + 1} / {frames.length}</span><button autoFocus onClick={onClose} aria-label="Close evidence"><X size={22} /></button></header>
    <EvidenceImage src={frame.src} revision={revision} label={frame.label} />
    <nav aria-label="Evidence navigation"><button disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button><button disabled={index === frames.length - 1} onClick={() => setIndex(index + 1)}>Next</button></nav>
  </dialog>;
}

export function IncidentDetailPage({ incidentId, timeZone }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [missing, setMissing] = useState(false);
  const [retry, setRetry] = useState(0);
  const [playbackIncident, setPlaybackIncident] = useState(null);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [playbackKey, setPlaybackKey] = useState(0);
  useEffect(() => {
    let alive = true;
    let busy = false;
    const controller = new AbortController();
    async function load() {
      if (busy || document.hidden) return;
      busy = true;
      try {
        const response = await fetch(`/api/incidents/notification/${encodeURIComponent(incidentId)}`, { signal: controller.signal });
        if (response.status === 404) {
          if (alive) { setMissing(true); setData(null); setError(""); }
          return;
        }
        if (!response.ok) throw new Error("Incident request failed");
        const result = await response.json();
        if (alive) { setData(result); setError(""); setMissing(false); }
      } catch (failure) {
        if (alive && failure.name !== "AbortError") setError("Could not refresh this incident. Check your connection and try again.");
      } finally { busy = false; }
    }
    void load();
    const timer = window.setInterval(load, 5000);
    document.addEventListener("visibilitychange", load);
    window.addEventListener("online", load);
    return () => {
      alive = false;
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", load);
      window.removeEventListener("online", load);
    };
  }, [incidentId, retry]);
  const incident = data?.incident;
  const notification = data?.notification;
  const ongoing = ["new", "updated"].includes(notification?.state);
  const events = useMemo(() => [...(incident?.events || [])].sort((a, b) => new Date(a.created_at) - new Date(b.created_at)), [incident?.events]);
  const representative = events.some((event) => Number(event.id) === Number(notification?.representative_event_id))
    ? notification.representative_event_id : incident?.representative_event_id;
  const evidenceFrames = useMemo(() => {
    const seen = new Set();
    return events.flatMap((event) => incidentEvidenceFrames(event).map((frame) => {
      const src = frame.kind === "snapshot"
        ? `/api/events/${event.id}/thumbnail.jpg?width=1280&quality=85&revision=${encodeURIComponent(notification?.revision || incident?.end_at || 0)}`
        : `/api/cameras/${encodeURIComponent(incident.camera_id)}/recordings/preview.jpg?epoch=${encodeURIComponent(frame.epoch)}&source=main&width=1280&exact=true`;
      return { ...frame, src, label: `${frame.label} · ${formatDateTime(new Date(frame.epoch * 1000).toISOString(), timeZone)}` };
    })).filter((frame) => {
      if (seen.has(frame.src)) return false;
      seen.add(frame.src);
      return true;
    });
  }, [events, incident?.camera_id, incident?.end_at, notification?.revision, timeZone]);
  const labels = notification?.classes || incident?.labels || [];
  const people = notification?.people || (incident?.identities || []).map((item) => item.name).filter(Boolean);
  const zones = notification?.zones || incident?.zones || [];
  const subject = [...people, ...labels.filter((label) => !people.length || label !== "person")].join(", ") || "Motion";
  const summary = notification?.summary || `${subject.charAt(0).toUpperCase()}${subject.slice(1)} detected at ${data?.camera_name || "this camera"}.`;
  const revision = notification?.revision || incident?.end_at;
  const status = ongoing ? "Ongoing" : notification?.state === "complete" ? "Completed" : "Recorded";
  const showTime = (value) => value ? formatDateTime(value, timeZone) : "";
  return <main className="incident-detail-page"><div className="incident-detail-container">
    <header className="incident-detail-nav"><a href={appUrl("/incidents")}><ArrowLeft size={18} />All incidents</a><span>SurvNG</span></header>
    {error ? <div className="incident-detail-notice" role="alert"><span>{error}{data ? " Showing the last received details." : ""}</span><button onClick={() => setRetry((value) => value + 1)}><RefreshCw size={16} />Retry</button></div> : null}
    {missing ? <section className="incident-detail-empty"><ImageOff size={36} /><h1>Incident unavailable</h1><p>The incident may have expired or been removed.</p><button onClick={() => setRetry((value) => value + 1)}>Try again</button></section>
      : !data ? (!error && <section className="incident-detail-empty" aria-busy="true">Loading incident…</section>)
        : <>
          <section className="incident-detail-heading">
            <div className="incident-detail-eyebrow"><span>{data.camera_name}</span><span className={ongoing ? "incident-detail-status ongoing" : "incident-detail-status"}>{status}</span></div>
            <h1>{summary}</h1>
            <p><time dateTime={notification?.started_at || incident.start_at}>{showTime(notification?.started_at || incident.start_at)}</time>{zones.length ? ` · ${zones.join(", ")}` : ""}</p>
          </section>
          <section className="incident-detail-media" aria-label="Incident evidence">
            {playbackIncident ? <div className="incident-detail-player"><IncidentClipLayer key={playbackKey} event={playbackIncident} active /><button className="incident-detail-close-player" onClick={() => setPlaybackIncident(null)} aria-label="Close playback"><X size={20} /></button></div>
              : <EvidenceImage eventId={representative} revision={revision} label={`Incident at ${data.camera_name}`} />}
          </section>
          <div className="incident-detail-actions">
            <button className="primary" disabled={!representative} onClick={() => { setSelectedFrame(null); setPlaybackIncident(incident); setPlaybackKey((value) => value + 1); }}><Play size={20} />{playbackIncident ? "Replay incident" : "Play incident"}</button>
            {ongoing ? <a href={appUrl(`/?camera=${encodeURIComponent(incident.camera_id)}`)}><Camera size={20} />Live view</a> : null}
          </div>
          <p className="incident-detail-hint">Playback includes the recording just before detection, when available.</p>
          <section className="incident-detail-section"><h2>Evidence</h2><div className="incident-detail-frames">
            {evidenceFrames.map((frame, index) => <EvidenceImage key={frame.src} src={frame.src} revision={revision} label={frame.label} onClick={() => setSelectedFrame({ frames: evidenceFrames, index })} />)}
          </div>{!evidenceFrames.length ? <p>No evidence images are available.</p> : <p className="incident-detail-hint">Tap an image to open it. Frames from expired recordings may be unavailable.</p>}</section>
          {selectedFrame ? <EvidenceViewer frames={selectedFrame.frames} selected={selectedFrame.index} revision={revision} onClose={() => setSelectedFrame(null)} /> : null}
          <section className="incident-detail-section"><h2>What happened</h2><ol className="incident-detail-timeline">
            {events.map((event) => <li key={event.id}><time dateTime={event.created_at}>{showTime(event.created_at)}</time><strong>{event.labels?.length ? `${event.labels.join(", ")} detected` : "Motion detected"}</strong>{event.zones?.length ? <span>{event.zones.join(", ")}</span> : null}</li>)}
            {people.length ? <li><strong>Recognized: {people.join(", ")}</strong></li> : null}
            {notification?.completed_at ? <li><time dateTime={notification.completed_at}>{showTime(notification.completed_at)}</time><strong>Incident completed</strong></li> : null}
            {!events.length ? <li>Detection details are unavailable.</li> : null}
          </ol></section>
          <footer className="incident-detail-footer"><a href={appUrl(`/incidents?event_ids=${representative}`)}>Open full investigation</a></footer>
        </>}
  </div></main>;
}
