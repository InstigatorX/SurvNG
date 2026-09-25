import React, { useEffect, useRef, useState } from "react";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";

export function VisitsPanel({ people, personId, timeZone }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [date, setDate] = useState("");
  const [hours, setHours] = useState(24);
  const [selection, setSelection] = useState([]);
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
  return <section className="person-visits" aria-label="Person visits">
    <div className="visit-controls">
      <label>Start (UTC; blank shows recent visits)<input type="datetime-local" value={date} disabled={busy} onChange={(event) => setDate(event.target.value)} /></label>
      <label>Window<select value={hours} disabled={busy} onChange={(event) => setHours(Number(event.target.value))}><option value={1}>1 hour</option><option value={6}>6 hours</option><option value={24}>24 hours</option></select></label>
      <button type="button" onClick={() => void load()} disabled={busy}>Refresh visits</button>
      {personName ? <span>Showing {personName}</span> : null}
    </div>
    <p>Names come from identified face sightings. Linked sightings belong to the same visit; they do not become trusted face references.</p>
    {data && !data.auto_link_enabled ? <p>Links require your review. Automatic linking is disabled until matching has been calibrated.</p> : null}
    {data && !data.routes_configured ? <p>Configure camera transition routes to receive cross-camera link suggestions. You can also select two sightings and confirm a link.</p> : null}
    {data?.truncated ? <p role="status">This window exceeds the 400-sighting limit. Results and links may be incomplete. Choose a shorter window; automatic linking is suspended for incomplete results.</p> : null}
    {data?.suggestions_truncated ? <p>Showing the first 200 link suggestions. Choose a shorter window to review more.</p> : null}
    {error ? <div role="alert">{error} <button type="button" onClick={() => void load()}>Retry</button></div> : null}
    {!data && !error ? <p role="status">Loading visits…</p> : null}
    {data && !visits.length ? <p>No person sightings in this window{personName ? ` for ${personName}` : ""}.</p> : null}
    <div className="visit-controls">
      <span>Select two sightings to review their relationship.</span>
      <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "accept")}>Same person and visit</button>
      <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "reject")}>Keep separate</button>
      <button type="button" disabled={busy || selection.length !== 2} onClick={() => void decide(...selection, "reset")}>Reset decision</button>
    </div>
    {visits.map((visit) => <article className="person-visit" key={visit.id}>
      <h3>{visit.person_name || "Unresolved person"} · {visit.sightings.length} sighting{visit.sightings.length === 1 ? "" : "s"}</h3>
      <p>{formatDateTime(visit.first_seen, timeZone)} — {formatDateTime(visit.last_seen, timeZone)}</p>
      <ul className="visit-sightings">{visit.sightings.map((item) => <li key={item.id}>
        <label><input type="checkbox" checked={selection.includes(item.id)} disabled={busy} onChange={() => toggle(item.id)} />{item.camera_id} · {formatDateTime(item.first_seen, timeZone)}</label>
        {item.face_id ? <img src={appUrl(`/api/faces/observations/${item.face_id}/crop.jpg`)} alt="Face evidence" /> : null}
        <strong>{({ confirmed: "Confirmed identity", recognized: "Automatically recognized", linked: `Linked to ${visit.person_name}'s visit`, unresolved: "Unresolved person", conflict: "Conflicting identity evidence" })[item.identity_status]}</strong>
        <a href={appUrl(`/incidents?event_ids=${item.event_id}`)}>View incident</a>
        <a href={recordingsHref({ cameraId: item.camera_id, epoch: Date.parse(item.first_seen) / 1000 })}>View recording</a>
        {item.face_id ? <a href={appUrl(`/people?face=${item.face_id}`)}>Review face</a> : null}
      </li>)}</ul>
      {visit.links.map((link) => <div className="visit-link" key={`${link.left}/${link.right}`}><span>{link.reason}{link.route ? ` · ${link.route} · ${link.gap_seconds}s` : ""}</span><button type="button" disabled={busy} onClick={() => void decide(link.left, link.right, "reject")}>Separate sightings</button></div>)}
    </article>)}
    {!personId && data?.suggestions?.length ? <section aria-label="Suggested visit links"><h3>Links to review</h3>{data.suggestions.map((link) => <div className="visit-link" key={`${link.left}/${link.right}`}>
      <span>{sightings.get(link.left)?.camera_id} ({formatDateTime(sightings.get(link.left)?.first_seen, timeZone)}) → {sightings.get(link.right)?.camera_id} ({formatDateTime(sightings.get(link.right)?.first_seen, timeZone)}) · {link.reason}{link.similarity != null ? ` · similarity ${link.similarity.toFixed(3)}` : ""}{link.ambiguous ? " · competing match" : ""}</span>
      <button type="button" disabled={busy || link.blocked} onClick={() => void decide(link.left, link.right, "accept")}>Confirm link</button>
      <button type="button" disabled={busy} onClick={() => void decide(link.left, link.right, "reject")}>Keep separate</button>
    </div>)}</section> : null}
  </section>;
}
