import React, { useEffect, useState } from "react";
import { appUrl, fetch } from "../shared/api.js";
import { formatTimeOnly } from "../shared/format.js";
import { recordingPreviewUrl } from "../shared/mediaUrls.js";
import { sceneEstablishmentText } from "../incidentScene.mjs";
import {
  cardOrder, coverageChips, cropBackgroundStyle, epochOf, establishmentReasonText, objectBestObservation, objectName, objectPath,
  objectSummary, observationSnapshotUrl,
} from "../incidentStory.mjs";
import { VisualSimilarIncidents } from "./IncidentCard.jsx";
import { IncidentAnalysisStatus } from "./IncidentAnalysisStatus.jsx";

function ObjectThumb({ object }) {
  const observation = objectBestObservation(object);
  const still = appUrl(observationSnapshotUrl(observation));
  const epoch = epochOf(observation?.captured_at);
  const previews = observation?.camera_id && Number.isFinite(epoch)
    ? ["main", "live"].map((source) => recordingPreviewUrl(encodeURIComponent(observation.camera_id), epoch, source, { width: 320, exact: true }))
    : [];
  const urls = [still, ...previews].filter(Boolean);
  const key = urls.join("\n");
  const [error, setError] = useState(null);
  const index = error?.key === key ? error.index : 0;
  const src = urls[index];
  const tall = object.kind === "person";
  const crop = src && src === still ? cropBackgroundStyle(observation, tall ? 36 / 42 : 36 / 34) : null;
  let style;
  if (crop) {
    const [width, height] = crop.backgroundSize.split(" ").map(parseFloat);
    const [x, y] = crop.backgroundPosition.split(" ").map(parseFloat);
    style = { width: `${width}%`, height: `${height}%`, left: `${(100 - width) * x / 100}%`, top: `${(100 - height) * y / 100}%` };
  }
  return <span className={`story-thumb${tall ? " tall" : ""}`} aria-hidden="true">
    {src ? <img key={src} src={src} alt="" loading="lazy" style={style} onError={() => setError({ key, index: index + 1 })} /> : null}
  </span>;
}

export function ObjectCards({ objects, selectedId, cameraName, timeZone, identities, objectNames = null, onSelect }) {
  return <div className="story-cards">
    {cardOrder(objects).map((object) => {
      const selected = String(object.id) === String(selectedId);
      const dimmed = selectedId != null && !selected;
      // Prefer a name bound to this track, then any unambiguous incident name.
      const bound = objectNames?.[object.id];
      const cardIdentities = bound?.length ? bound.map((name) => ({ name })) : bound ? (identities || []) : identities;
      return <button key={object.id} type="button" className={`story-card ${object.kind}${selected ? " selected" : ""}${dimmed ? " dimmed" : ""}`}
        aria-pressed={selected} onClick={() => onSelect(selected ? null : object.id)}>
        <ObjectThumb object={object} />
        <span className="story-card-body">
          <span className="story-card-title">{objectName(object)}{object.kind === "uncertain" ? <> <span className="story-uncertain-tag">uncertain</span></> : null}</span>
          <span className="story-card-summary">{objectSummary(object, { cameraName, timeZone, identities: cardIdentities })}</span>
        </span>
      </button>;
    })}
  </div>;
}

function alertChip(decisions) {
  if (!Array.isArray(decisions) || !decisions.length) return null;
  const eligible = decisions.flatMap((decision) => (decision.objects || []).filter((object) => object.eligible));
  if (!eligible.length && !decisions.some((decision) => decision.eligible)) return { tone: "neutral", text: "Alert criteria not met" };
  const first = eligible[0];
  const label = first ? `${objectName(first)}${first.zones?.length ? ` in ${first.zones[0]}` : ""}` : "";
  // Policy decisions do not confirm delivery, so say what was decided.
  return { tone: "info", text: `Met alert criteria${label ? ` · ${label}` : ""}` };
}

export function WhyPanel({ incident, cameraName, hasSightings }) {
  const establishment = sceneEstablishmentText(incident?.establishment);
  const reason = establishmentReasonText(incident, cameraName) || establishment?.explanation;
  const status = establishment?.status;
  const motionOnly = !hasSightings && (incident.has_objects === false || !incident.labels?.length);
  const heading = status && status !== "established" ? establishment.title : motionOnly ? "Motion only" : "Why this is an incident";
  const alert = alertChip(incident?.alert_decisions);
  const coverage = coverageChips(incident, cameraName);
  const hasGaps = incident?.coverage?.gaps?.length || incident?.episodes?.some((episode) => episode.coverage?.gaps?.length);
  return <section className="story-context" aria-label={heading}>
    <h2>{heading}</h2>
    {reason ? <p>{reason}</p> : null}
    {alert || hasGaps || coverage.length ? <div className="story-chips">
      {alert ? <span className={`story-chip ${alert.tone}`}>{alert.text}</span> : null}
      {coverage.map((chip) => <span key={chip.text} className={`story-chip ${chip.tone}`}>{chip.text}</span>)}
      {hasGaps && !coverage.length ? <span className="story-chip warn">Recording analysis has gaps</span> : null}
    </div> : null}
    {status && status !== "established" ? <a className="story-link" href={appUrl("/observations")}>Review observations</a> : null}
    <IncidentAnalysisStatus hideComplete key={incident.incident_id || incident.id} incidentId={incident.incident_id || incident.id} />
  </section>;
}

export function PathPanel({ object, cameraName, timeZone, activeCameraId, canCorrect, onCorrect, onOpenEvent }) {
  const [similar, setSimilar] = useState(false);
  useEffect(() => setSimilar(false), [object.id]);
  const steps = objectPath(object, cameraName);
  const activeIndex = Math.max(0, steps.findIndex((step) => step.cameraId === activeCameraId));
  const searchable = object.source_event_id && Number.isInteger(object.object_index);
  const title = `${objectName(object)} ${steps.length > 1 ? "· path across cameras" : "sighting"}`;
  const stepTime = (step) => `${formatTimeOnly(step.start, timeZone)}${step.end > step.start ? ` – ${formatTimeOnly(step.end, timeZone)}` : ""}${step.zones.length ? ` · ${step.zones.join(", ")} zone` : ""}`;
  return <section className="story-context" aria-label={title}>
    <h2>{title}</h2>
    {searchable || canCorrect ? <div className="story-actions">
      {searchable ? <button type="button" aria-pressed={similar} onClick={() => setSimilar((current) => !current)}>Find similar</button> : null}
      {canCorrect ? <button type="button" onClick={onCorrect}>Correct this object</button> : null}
    </div> : null}
    {steps.length === 1 ? <p className="story-single-step">{steps[0].camera} · {stepTime(steps[0])}</p> : null}
    {steps.length > 1 ? <ol className="story-path">
      {steps.map((step, index) => <li key={`${step.cameraId}-${index}`} className={index === activeIndex ? "active" : ""} aria-current={index === activeIndex ? "step" : undefined}>
        <div className="story-step"><span className="story-step-number">{step.number}</span>
          <span className="story-step-text"><strong>{step.camera}</strong><small>{stepTime(step)}</small></span></div>
        {step.note ? <p className="story-step-note">{step.note}</p> : null}
      </li>)}
    </ol> : null}
    {similar ? <VisualSimilarIncidents active anchorEventId={Number(object.source_event_id)} objectIndex={object.object_index} objectLabel={object.label}
      trackId={objectBestObservation(object)?.track_id ?? null} cameraNameById={{ get: cameraName }} timeZone={timeZone}
      onSelect={(row) => onOpenEvent?.(row?.id ?? row?.event_id)} onClear={() => setSimilar(false)} /> : null}
  </section>;
}

export function CorrectPanel({ incident, object, objects, cameraName, timeZone, canCorrect, onCorrected, onDone }) {
  const [mode, setMode] = useState("");
  const [label, setLabel] = useState(object.label || "");
  const [other, setOther] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  // Reset only when another object is chosen; a saved correction changes
  // this object's label and must keep its confirmation visible.
  useEffect(() => { setMode(""); setLabel(object.label || ""); setOther(""); setError(""); setNotice(""); }, [object.id]);
  const cameras = [...new Set(object.intervals.map((interval) => cameraName(interval.camera_id)))];
  const count = object.observation_count || object.observations?.length || 0;
  const others = objects.filter((item) => item.id !== object.id);
  async function submit(body, done, keepId = null) {
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await fetch(`/api/incidents/${encodeURIComponent(incident.incident_id)}/corrections`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_revision: incident.revision, ...body }),
      });
      if (response.status === 409) throw new Error("This incident changed. Its latest evidence is loading; review it and try again.");
      if (!response.ok) throw new Error("Could not save the correction. The evidence has not been changed.");
      const detail = await response.json();
      setMode(""); setNotice(done);
      onCorrected?.(detail, keepId);
    } catch (failure) { setError(failure.message); }
    finally { setBusy(false); }
  }
  const correctable = canCorrect && incident?.incident_id && Number.isFinite(Number(incident?.revision)) && !String(object.id).startsWith("historical-");
  return <section className={`story-context${object.kind === "uncertain" ? " uncertain" : ""}`} aria-label="Correct this object">
    <h2>Correct this object</h2>
    <p>{count === 1 ? "One supporting image" : `${count} supporting images`} from {cameras.join(", ") || "this incident"}.{object.kind === "uncertain" ? " Detector confidence is low, so the label is shown as uncertain." : ""}</p>
    {correctable ? <>
      <div className="story-actions">
        {object.kind === "uncertain" ? <button type="button" className="confirm" disabled={busy} onClick={() => submit({ operation: "label", object_id: object.id, label: object.label }, "Label confirmed.")}>Yes, a {String(object.label || "object").replaceAll("_", " ")}</button> : null}
        <button type="button" aria-expanded={mode === "label"} onClick={() => setMode(mode === "label" ? "" : "label")}>Change label</button>
        {others.length ? <button type="button" aria-expanded={mode === "associate"} onClick={() => setMode(mode === "associate" ? "" : "associate")}>Same as another object</button> : null}
      </div>
      {mode === "label" ? <form className="story-correct-form" onSubmit={(event) => { event.preventDefault(); if (label.trim()) void submit({ operation: "label", object_id: object.id, label: label.trim() }, "Label corrected."); }}>
        <label>New label<input value={label} maxLength={100} onChange={(event) => setLabel(event.target.value)} required /></label>
        <button type="submit" disabled={busy}>{busy ? "Saving…" : "Save label"}</button>
      </form> : null}
      {mode === "associate" ? <form className="story-correct-form" onSubmit={(event) => { event.preventDefault(); if (other) void submit({ operation: "associate", object_ids: [other, object.id] }, "Objects associated.", other); }}>
        <label>Same object as<select value={other} onChange={(event) => setOther(event.target.value)} required>
          <option value="">Choose an object</option>
          {others.map((item) => <option key={item.id} value={item.id}>{objectName(item)} · {formatTimeOnly(item.intervals[0]?.start, timeZone)}</option>)}
        </select></label>
        <button type="submit" disabled={busy}>{busy ? "Saving…" : "Associate"}</button>
      </form> : null}
    </> : <p className="story-muted">{canCorrect ? "This historical record cannot be corrected here." : "Only an administrator can correct objects."}</p>}
    {error ? <p role="alert" className="story-error">{error}</p> : null}
    {notice ? <p role="status" className="story-muted">{notice}</p> : null}
    <span className="story-muted small">Corrections keep the original observations and old incident links.</span>
    {object.kind !== "uncertain" && onDone ? <button type="button" className="story-link-button" onClick={onDone}>Back to path</button> : null}
  </section>;
}
