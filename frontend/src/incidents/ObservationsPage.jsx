import React, { useEffect, useState } from "react";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "../shared/datetime.js";
import { formatDateTime } from "../shared/format.js";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { canonicalIncidentHref, sceneCoverageText, sceneObjectLabel } from "../incidentScene.mjs";
import { ObservationImage } from "./IncidentScenePanel.jsx";
import "./observations.css";

const statuses = {
  pending: "Awaiting confirmation",
  not_established: "Activity not established",
  incomplete: "Analysis incomplete",
  established: "Established incident",
};
const explanations = {
  pending: "Further evidence is being evaluated. An incident has not yet been established.",
  not_established: "These observations did not establish an episode of activity.",
  incomplete: "Analysis could not be completed. This does not establish that nothing happened.",
  established: "These observations are linked to an established incident.",
};

export function ObservationsPage({ timeZone }) {
  const initial = new URLSearchParams(window.location.search);
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const [day, setDay] = useState(/^\d{4}-\d{2}-\d{2}$/.test(initial.get("day") || "") && Number.isFinite(Date.parse(`${initial.get("day")}T00:00:00Z`)) ? initial.get("day") : today);
  const [status, setStatus] = useState(statuses[initial.get("status")] ? initial.get("status") : "all");
  const [camera, setCamera] = useState(initial.get("camera_id") || "");
  const [selectedId, setSelectedId] = useState(initial.get("observation_id") || "");
  const [offset, setOffset] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [cameras, setCameras] = useState([]);
  const [cameraError, setCameraError] = useState(false);
  const [page, setPage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [selectedObservation, setSelectedObservation] = useState(null);
  const cameraName = (id) => cameras.find((item) => item.id === id)?.name || id || "Camera";
  const date = (value) => value ? formatDateTime(value, timeZone) : "Time unknown";
  useVisiblePolling(() => setRefresh((value) => value + 1), 20000, true, { immediate: false });
  useEffect(() => {
    const online = () => setRefresh((value) => value + 1);
    window.addEventListener("online", online);
    return () => window.removeEventListener("online", online);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal }).then((response) => { if (!response.ok) throw new Error("Camera list unavailable"); return response.json(); }).then((items) => { if (!controller.signal.aborted) { setCameras(Array.isArray(items) ? items : []); setCameraError(false); } }).catch(() => { if (!controller.signal.aborted) setCameraError(true); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const url = new URL(window.location.href);
    for (const [key, value] of Object.entries({ day, status: status === "all" ? "" : status, camera_id: camera, observation_id: selectedId })) {
      if (value) url.searchParams.set(key, value); else url.searchParams.delete(key);
    }
    window.history.replaceState(window.history.state, "", url);
  }, [day, status, camera, selectedId]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    const params = new URLSearchParams({
      start_at: new Date(zonedDateSecondToEpoch(day, 0, timeZone) * 1000).toISOString(),
      end_at: new Date(zonedDateSecondToEpoch(addDaysToDateKey(day, 1), 0, timeZone) * 1000).toISOString(),
      limit: "25", offset: String(offset),
    });
    if (status !== "all") params.set("status", status);
    if (camera) params.set("camera_id", camera);
    fetch(`/api/observations?${params}`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error("Observations could not be loaded. Try refreshing.");
      const result = await response.json();
      if (!controller.signal.aborted) setPage(result);
    }).catch((failure) => { if (!controller.signal.aborted) setError(failure.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [day, status, camera, offset, refresh, timeZone]);
  useEffect(() => {
    if (!selectedId) { setDetail(null); setDetailError(""); setDetailLoading(false); return undefined; }
    const controller = new AbortController();
    setDetailLoading(true); setDetailError("");
    fetch(`/api/observations/${encodeURIComponent(selectedId)}`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error(response.status === 404 ? "This observation record is no longer available." : "Supporting observations could not be loaded. Try refreshing.");
      const result = await response.json();
      if (!controller.signal.aborted) setDetail(result);
    }).catch((failure) => { if (!controller.signal.aborted) setDetailError(failure.message); }).finally(() => { if (!controller.signal.aborted) setDetailLoading(false); });
    return () => controller.abort();
  }, [selectedId, refresh]);
  function changeFilter(setter, value) {
    setter(value); setOffset(0); setPage(null); setSelectedId(""); setSelectedObservation(null);
  }
  function select(id) { setDetail(null); setSelectedObservation(null); setSelectedId(String(id)); }
  const observations = detail?.observations || [];
  const evidence = observations.find((item, index) => String(item.id ?? index) === selectedObservation);
  return <main className="observations-page">
    <header><div><h1>Observations</h1><p>Retained evidence before and after activity is established. These records are not all incidents.</p></div><a href={appUrl("/review")}>Review</a></header>
    <form className="observations-filters" onSubmit={(event) => event.preventDefault()}>
      <label>Day<input aria-label="Day" type="date" value={day} onChange={(event) => { if (event.target.value) changeFilter(setDay, event.target.value); }} /></label>
      <label>Status<select aria-label="Status" value={status} onChange={(event) => changeFilter(setStatus, event.target.value)}><option value="all">All observations</option>{Object.entries(statuses).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
      <label>Camera<select aria-label="Camera" value={camera} onChange={(event) => changeFilter(setCamera, event.target.value)}><option value="">All cameras</option>{camera && !cameras.some((item) => item.id === camera) ? <option value={camera}>{camera}</option> : null}{cameras.map((item) => <option value={item.id} key={item.id}>{item.name || item.id}</option>)}</select></label>
      <button type="button" onClick={() => setRefresh((value) => value + 1)}>Refresh observations</button>
    </form>
    {cameraError ? <p>Camera names are unavailable; retained camera IDs are shown.</p> : null}
    {loading ? <p role="status">Loading observations…</p> : null}
    {error ? <p role="alert">{error}{page ? " Previously loaded records remain visible." : ""}</p> : null}
    {!loading && !error && !page?.items?.length ? <p>No observations match this day and these filters. This does not establish complete camera coverage.</p> : null}
    <div className={`observations-layout${selectedId ? " has-detail" : ""}`}><section aria-label="Observation records">
      <ul className="observations-list">{(page?.items || []).map((item) => <li key={item.id}>
        <button type="button" aria-pressed={selectedId === String(item.id)} onClick={() => select(item.id)}><strong>{statuses[item.status] || "Status unknown"}</strong><span>{cameraName(item.camera_id)} · {date(item.captured_at)}</span><span>{item.summary || explanations[item.status] || "Retained observations"}</span></button>
        {item.incident_id && item.status === "established" ? <a href={appUrl(canonicalIncidentHref(item))}>Open established incident</a> : null}
      </li>)}</ul>
      {page ? <nav aria-label="Observation pages"><button type="button" disabled={offset === 0 || loading} onClick={() => { setPage(null); setOffset((value) => Math.max(0, value - 25)); }}>Previous</button><span>Page {Math.floor(offset / 25) + 1}</span><button type="button" disabled={loading || offset + 25 >= Number(page.total || 0)} onClick={() => { setPage(null); setOffset((value) => value + 25); }}>Next</button></nav> : null}
    </section>
    {selectedId ? <section className="observation-detail" aria-label="Observation details">
      <button type="button" onClick={() => setSelectedId("")}>Close observation details</button>
      {detailLoading ? <p role="status">Loading supporting observations…</p> : null}{detailError ? <p role="alert">{detailError}</p> : null}
      {detail ? <><h2>{statuses[detail.status] || "Observation record"}</h2><p>{explanations[detail.status]}</p><p>{cameraName(detail.camera_id)} · {date(detail.captured_at)}</p>
        {detail.summary ? <p>{detail.summary}</p> : null}{detail.reason ? <details><summary>Evaluation details</summary><p>{String(detail.reason).replaceAll("_", " ")}</p></details> : null}
        <p>{sceneCoverageText(detail.coverage)}</p>
        {detail.incident_id ? <a href={appUrl(canonicalIncidentHref(detail))}>{detail.status === "established" ? "Open established incident" : "Open preserved incident record"}</a> : null}
        <details><summary>Retained object observations ({observations.length})</summary><ul className="observation-evidence-list">{observations.map((item, index) => <li key={item.id ?? index}><button type="button" aria-pressed={String(item.id ?? index) === selectedObservation} onClick={() => setSelectedObservation(String(item.id ?? index))}>{sceneObjectLabel({ ...item, certainty: item.certainty || "possible" })} · {date(item.captured_at)}</button></li>)}</ul>{!observations.length ? <p>No object candidates were retained for this record.</p> : null}</details>
        {evidence ? <><ObservationImage observation={evidence} timeZone={timeZone} /><details><summary>Detection details</summary><p>Detector confidence: {Math.round(Number(evidence.confidence || 0) * 100)}%</p></details>
          {evidence.camera_id && Number.isFinite(Date.parse(evidence.captured_at)) ? <a href={recordingsHref({ cameraId: evidence.camera_id, epoch: Date.parse(evidence.captured_at) / 1000, source: "main" })}>Open recording at observation time</a> : null}</> : null}
      </> : null}
    </section> : null}</div>
  </main>;
}
