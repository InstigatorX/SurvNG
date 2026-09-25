import React, { useEffect, useRef, useState } from "react";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";

function SightingPreview({ item }) {
  const [preview, setPreview] = useState(item.face_id ? "face" : "snapshot");
  if (preview === "face") return <img
    src={appUrl(`/api/faces/observations/${item.face_id}/crop.jpg`)}
    alt="Face evidence" loading="lazy" onError={() => setPreview("snapshot")}
  />;
  return <figure className="visit-snapshot-fallback">
    {preview === "snapshot" ? <img
      className="visit-incident-snapshot"
      src={appUrl(`/api/events/${item.event_id}/thumbnail.jpg?width=520&quality=82&object_focus=false`)}
      alt={`Incident snapshot at ${item.camera_id}`} loading="lazy"
      onError={() => setPreview("unavailable")}
    /> : <span className="visit-snapshot-unavailable">Snapshot unavailable · View recording below</span>}
    <figcaption>{item.face_id ? "Face crop unavailable" : "No face captured"} · Incident snapshot</figcaption>
  </figure>;
}

function SightingEvidence({ item, personName, timeZone }) {
  return <>
    <span>{item.camera_id} · {formatDateTime(item.first_seen, timeZone)}</span>
    <SightingPreview key={`${item.event_id}:${item.face_id || "none"}`} item={item} />
    <strong>{({ confirmed: "Confirmed identity", recognized: "Automatically recognized", linked: `Linked to ${personName}'s visit`, unresolved: "Unresolved person", conflict: "Conflicting identity evidence" })[item.identity_status]}</strong>
    <a href={appUrl(`/incidents?event_ids=${item.event_id}`)}>View incident</a>
    <a href={recordingsHref({ cameraId: item.camera_id, epoch: Date.parse(item.first_seen) / 1000 })}>View recording</a>
    {item.face_id ? <a href={appUrl(`/people?face=${item.face_id}`)}>Review face</a> : null}
  </>;
}

export function VisitsPanel({ people, personId, timeZone }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [date, setDate] = useState("");
  const [hours, setHours] = useState(24);
  const [selection, setSelection] = useState([]);
  const [manualReview, setManualReview] = useState(false);
  const sequence = useRef(0);
  const request = useRef(null);
  const mounted = useRef(false);
  const dateRef = useRef(date);
  const hoursRef = useRef(hours);
  dateRef.current = date;
  hoursRef.current = hours;

  async function load() {
    const serial = ++sequence.current;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const start = dateRef.current ? Date.parse(`${dateRef.current}Z`) / 1000 : Date.now() / 1000 - hoursRef.current * 3600;
    const query = `?start=${start}&end=${start + hoursRef.current * 3600}`;
    try {
      const response = await fetch(`/api/people/visits${query}`, { signal: controller.signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not load visits");
      if (serial !== sequence.current || !mounted.current) return;
      setData(payload);
      setError("");
    } catch (failure) {
      if (serial === sequence.current && mounted.current && failure.name !== "AbortError") setError(failure.message || "Could not load visits");
    }
  }

  useEffect(() => {
    mounted.current = true;
    setData(null);
    setSelection([]);
    void load();
    const timer = setInterval(() => { if (!document.hidden) void load(); }, 15000);
    const reconnect = () => void load();
    window.addEventListener("online", reconnect);
    return () => {
      mounted.current = false;
      sequence.current += 1;
      request.current?.abort();
      clearInterval(timer);
      window.removeEventListener("online", reconnect);
    };
  }, [date, hours]);

  const sightings = new Map((data?.visits || []).flatMap((visit) => visit.sightings.map((item) => [item.id, item])));
  const personName = people.find((person) => String(person.id) === String(personId))?.name;

  async function decide(leftId, rightId, decision) {
    const left = sightings.get(leftId), right = sightings.get(rightId);
    if (!left || !right || !data) return;
    setBusy(true);
    try {
      const response = await fetch("/api/people/visits/link", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ left_id: leftId, right_id: rightId, left_revision: left.revision, right_revision: right.revision, decision, start: data.start, end: data.end }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not update this link");
      if (!mounted.current) return;
      setSelection([]);
      await load();
    } catch (failure) {
      if (mounted.current) setError(failure.message || "Could not update this link");
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  function toggle(id) {
    setSelection((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current.slice(-1), id]);
  }

  const visits = (data?.visits || []).filter((visit) => !personId || String(visit.person_id) === String(personId));
  const visibleIds = new Set(visits.flatMap((visit) => visit.sightings.map((item) => item.id)));
  const suggestions = (data?.suggestions || []).filter((link) => sightings.has(link.left) && sightings.has(link.right)
    && (!personId || visibleIds.has(link.left) || visibleIds.has(link.right)));
  const visitNames = new Map((data?.visits || []).flatMap((visit) => visit.sightings.map((item) => [item.id, visit.person_name])));

  useEffect(() => {
    setSelection((current) => current.filter((id) => (data?.visits || []).some((visit) =>
      (!personId || String(visit.person_id) === String(personId)) && visit.sightings.some((item) => item.id === id))));
  }, [data, personId]);

  return <section className="person-visits" aria-label="Person visits">
    <h3>Group sightings into visits</h3>
    <p>Group sightings from one continuous visit, such as someone moving between cameras. Separate arrivals should stay separate, even for the same person.</p>
    <p>To confirm or correct an identity, choose <strong>Review face</strong> on a sighting. Linking visits does not confirm identities or add trusted face references.</p>
    <div className="visit-controls">
      <label>Start (UTC; blank shows recent visits)<input type="datetime-local" value={date} disabled={busy} onChange={(event) => setDate(event.target.value)} /></label>
      <label>Window<select value={hours} disabled={busy} onChange={(event) => setHours(Number(event.target.value))}><option value={1}>1 hour</option><option value={6}>6 hours</option><option value={24}>24 hours</option></select></label>
      <button type="button" onClick={() => void load()} disabled={busy}>Refresh visits</button>
      {personName ? <span>Showing {personName}</span> : null}
    </div>
    {data && !data.auto_link_enabled ? <p>Automatic linking is off. Suggested links only join visits after you confirm them.</p> : null}
    {data && !data.routes_configured ? <p>Configure camera transition routes to receive cross-camera link suggestions. Manual linking is also available below.</p> : null}
    {data?.truncated ? <p role="status">This window exceeds the 400-sighting limit. Results and links may be incomplete. Choose a shorter window; automatic linking is suspended for incomplete results.</p> : null}
    {data?.suggestions_truncated ? <p>Showing the first 200 link suggestions. Choose a shorter window to review more.</p> : null}
    {error ? <div role="alert">{error} <button type="button" onClick={() => void load()}>Retry</button></div> : null}
    {!data && !error ? <p role="status">Loading visits…</p> : null}
    {data && !visits.length ? <p>No person sightings in this window{personName ? ` for ${personName}` : ""}.</p> : null}
    {data ? <section className="visit-suggestions" aria-label="Suggested visit links">
      <h3>Suggested pairs to review{suggestions.length ? ` (${suggestions.length})` : ""}</h3>
      {suggestions.length ? <p>Review the snapshots, times and recordings. Confirm only if both sightings belong to the same person during one visit.</p>
        : <p role="status">No suggested links to review{personName ? ` for ${personName}` : ""}. The visit history below does not need a decision on every sighting.</p>}
      {suggestions.map((link) => <article className="visit-suggestion" key={`${link.left}/${link.right}`}>
        <ul className="visit-sightings">{[link.left, link.right].map((id) => <li key={id}>
          <strong>{visitNames.get(id) || "Unresolved person"}</strong>
          <SightingEvidence item={sightings.get(id)} personName={visitNames.get(id)} timeZone={timeZone} />
        </li>)}</ul>
        <p>{link.reason}{link.route ? ` · ${link.route}` : ""}{link.gap_seconds != null ? ` · ${link.gap_seconds}s between sightings` : ""}{link.ambiguous ? " · Competing match: check carefully" : ""}</p>
        <div className="visit-controls">
          <button type="button" disabled={busy || link.blocked} onClick={() => void decide(link.left, link.right, "accept")}>Same person and visit</button>
          <button type="button" disabled={busy} onClick={() => void decide(link.left, link.right, "reject")}>Keep separate</button>
        </div>
      </article>)}
    </section> : null}
    {data && visits.length ? <>
      <h3>Visit history</h3>
      <details className="visit-manual-review" open={manualReview} onToggle={(event) => {
        setManualReview(event.currentTarget.open);
        if (!event.currentTarget.open) setSelection([]);
      }}>
        <summary>Manually link or separate sightings</summary>
        <p>Select two sightings in the history below. Use this only to correct a visit grouping. Separate visits can stay as they are.</p>
        <div className="visit-controls">
          <span>{selection.length} of 2 sightings selected</span>
          <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "accept")}>Same person and visit</button>
          <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "reject")}>Keep separate</button>
          <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "reset")}>Reset decision</button>
        </div>
        <p>Reset decision removes your previous linking or separation decision for the selected pair.</p>
      </details>
    </> : null}
    {visits.map((visit) => <article className="person-visit" key={visit.id}>
      <h3>{visit.person_name || "Unresolved person"} · {visit.sightings.length} sighting{visit.sightings.length === 1 ? "" : "s"}</h3>
      <p>{formatDateTime(visit.first_seen, timeZone)} — {formatDateTime(visit.last_seen, timeZone)}</p>
      <ul className="visit-sightings">{visit.sightings.map((item) => <li key={item.id}>
        {manualReview ? <label><input type="checkbox" checked={selection.includes(item.id)} disabled={busy} onChange={() => toggle(item.id)} />Select sighting at {item.camera_id} · {formatDateTime(item.first_seen, timeZone)}</label> : null}
        <SightingEvidence item={item} personName={visit.person_name} timeZone={timeZone} />
      </li>)}</ul>
      {visit.links.map((link) => <div className="visit-link" key={`${link.left}/${link.right}`}><span>{link.reason}{link.route ? ` · ${link.route} · ${link.gap_seconds}s` : ""}</span><button type="button" disabled={busy} onClick={() => void decide(link.left, link.right, "reject")}>Separate sightings</button></div>)}
    </article>)}
  </section>;
}
