import React, { useEffect, useState } from "react";
import { appUrl, fetch, mediaUrl, recordingsHref } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { incidentEpisodeClips, incidentSceneObjects, observationBoxStyle, observationImageDescription, sceneActivityText, sceneCoverageText, sceneEstablishmentText, sceneNotificationSummary, sceneObjectLabel } from "../incidentScene.mjs";
import "./incident-scene.css";

export function ObservationImage({ observation, timeZone }) {
  const [failed, setFailed] = useState(false);
  const [reviewFailed, setReviewFailed] = useState(false);
  useEffect(() => { setFailed(false); setReviewFailed(false); }, [observation.id, observation.snapshot_url, observation.snapshot_available, observation.review_image?.url]);
  const style = observation.image?.analyzed_frame === false ? null : observationBoxStyle(observation);
  const snapshotUrl = mediaUrl(observation.snapshot_url) || (observation.id ? appUrl(`/api/incidents/observations/${encodeURIComponent(observation.id)}/snapshot`) : "");
  const available = snapshotUrl && (observation.snapshot_available === true || Boolean(observation.snapshot_url) && observation.snapshot_available !== false) && !failed;
  const review = observation.review_image;
  const reviewUrl = mediaUrl(review?.url);
  const date = (value) => value ? formatDateTime(value, timeZone) : "Capture time unknown";
  return <div className="incident-observation-evidence">
    {available ? <figure><div className="incident-observation-image">
      <img src={snapshotUrl} alt={`Retained evidence for ${observation.label || "object"} observation`} onError={() => setFailed(true)} />
      {style ? <span className="incident-observation-box" style={style} /> : null}
    </div><figcaption>{observationImageDescription(observation.image)}<br />{date(observation.image?.captured_at)}<br /><a href={snapshotUrl} target="_blank" rel="noopener noreferrer">Open retained image at original size</a></figcaption></figure>
      : <p>Supporting image unavailable. The retained observation remains available.</p>}
    {reviewUrl ? <details className="incident-review-image"><summary>Additional review image</summary>
      <p>This image provides additional context. Detection boxes from the analyzed frame are not applied to it.</p>
      {reviewFailed ? <p>Additional review image unavailable.</p> : <img src={reviewUrl} alt="Additional scene context, not the analyzed frame" onError={() => setReviewFailed(true)} />}
      <p>{observationImageDescription({ ...review, analyzed_frame: false })}<br />{date(review.captured_at)}</p>
    </details> : null}
  </div>;
}

export function IncidentScenePanel({ incident, timeZone, cameraNameById = new Map(), onSelectObservation, onSelectEpisode, onPlayScene, activeEpisodeId, onNextEpisode, onStopPlayback, onFindSimilar, onChanged }) {
  const [corrected, setCorrected] = useState(null);
  const [selectedObjectId, setSelectedObjectId] = useState(null);
  const [selectedObservationId, setSelectedObservationId] = useState(null);
  const [operation, setOperation] = useState("label");
  const [label, setLabel] = useState("");
  const [selectedIds, setSelectedIds] = useState([]);
  const [mergeIds, setMergeIds] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [allActivity, setAllActivity] = useState(false);
  useEffect(() => {
    setCorrected(null); setSelectedObjectId(null); setSelectedObservationId(null);
    setSelectedIds([]); setError(""); setNotice("");
    setAllActivity(false);
  }, [incident?.incident_id || incident?.id]);
  const scene = corrected && Number(corrected.revision) >= Number(incident?.revision || 0) ? corrected : incident;
  const objects = incidentSceneObjects(scene);
  const establishment = sceneEstablishmentText(scene?.establishment);
  const establishmentEvidence = objects.flatMap((object) => (object.observations || []).filter((observation) => scene?.establishment?.supporting_observation_ids?.includes(observation.id)).map((observation) => ({ object, observation })));
  const episodes = incidentEpisodeClips(scene);
  const activity = [...(scene?.activity || [])].sort((left, right) => Date.parse(left.captured_at) - Date.parse(right.captured_at));
  const selectedObject = objects.find((object) => String(object.id) === String(selectedObjectId));
  const observations = selectedObject?.observations || [];
  const selectedObservation = observations.find((observation, index) => String(observation.id ?? index) === String(selectedObservationId));
  const cameraName = (id) => cameraNameById.get(id) || id || "Camera";
  const date = (value) => value ? formatDateTime(value, timeZone) : "Time unknown";
  const correctionItems = operation === "split" ? (scene?.episodes || []).map((episode) => ({ id: episode.id, label: `${cameraName(episode.camera_id)} · ${date(episode.start_at)}` }))
    : operation === "separate" ? observations.map((observation) => ({ id: observation.id, label: `${cameraName(observation.camera_id)} · ${date(observation.captured_at)}` })).filter((item) => item.id != null)
      : objects.map((object) => ({ id: object.id, label: `${object.label} · ${date(object.first_seen_at)}` }));
  async function correct(event) {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const body = { expected_revision: scene.revision, operation };
      if (operation === "label") {
        if (!selectedObject || !label.trim()) throw new Error("Select an object and enter its label.");
        Object.assign(body, { object_id: selectedObject.id, label: label.trim() });
      } else if (operation === "merge") {
        const ids = [...new Set(mergeIds.split(/[\s,]+/).filter(Boolean))].filter((id) => id !== String(scene.incident_id));
        if (!ids.length) throw new Error("Enter another incident ID to merge.");
        const revisions = { [scene.incident_id]: scene.revision };
        const resolvedIds = [scene.incident_id];
        for (const id of ids) {
          const response = await fetch(`/api/incidents/detail?incident_id=${encodeURIComponent(id)}`);
          if (!response.ok) throw new Error(`Could not load incident ${id}.`);
          const detail = await response.json();
          if (!detail.incident_id || detail.revision == null) throw new Error(`Incident ${id} has no current revision.`);
          revisions[detail.incident_id] = detail.revision;
          resolvedIds.push(detail.incident_id);
        }
        Object.assign(body, { incident_ids: [...new Set(resolvedIds)], expected_revisions: revisions });
      } else {
        if (!selectedIds.length || (operation === "associate" && selectedIds.length < 2)) throw new Error(operation === "associate" ? "Select at least two objects." : "Select the evidence to separate.");
        body[operation === "split" ? "episode_ids" : operation === "associate" ? "object_ids" : "observation_ids"] = selectedIds;
      }
      const response = await fetch(`/api/incidents/${encodeURIComponent(scene.incident_id)}/corrections`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (response.status === 409) throw new Error("This incident changed. Refresh it and review the latest evidence before applying the correction.");
      if (!response.ok) throw new Error("Could not save the correction. Your evidence has not been changed here.");
      const detail = await response.json();
      setCorrected(detail); setSelectedIds([]); setNotice("Correction saved. Original observations are preserved."); onChanged?.(detail);
    } catch (failure) { setError(failure.message); }
    finally { setBusy(false); }
  }
  return <section className="incident-scene-panel" aria-label="Incident scene">
    {establishment ? <section className={`incident-establishment ${establishment.status}`} aria-label="Activity establishment">
      <strong>{establishment.title}</strong><p>{establishment.explanation}</p>
      {establishment.status !== "established" ? <a href={appUrl("/observations")}>Review observations</a> : null}
      {establishment.summary ? <p>{establishment.summary}</p> : null}
      {scene.establishment.reason || establishmentEvidence.length ? <details><summary>Establishment evidence</summary>
        {scene.establishment.reason ? <p>{String(scene.establishment.reason).replaceAll("_", " ")}</p> : null}
        {establishmentEvidence.map(({ object, observation }) => <button key={observation.id} type="button" onClick={() => { setSelectedObjectId(object.id); setSelectedObservationId(observation.id); setSelectedIds([]); setLabel(object.label); onSelectObservation?.(observation); }}>{sceneObjectLabel(object)} · {cameraName(observation.camera_id)} · {date(observation.captured_at)}</button>)}
      </details> : null}
    </section> : null}
    {scene?.summary && (!establishment || establishment.status === "established") ? <p className="incident-scene-summary">{scene.summary}</p> : null}
    {onPlayScene && episodes.some((episode) => episode.clip) ? <button type="button" onClick={onPlayScene}>{episodes.every((episode) => episode.clip) ? "Play whole incident" : "Play available episodes"}</button> : null}
    {episodes.length ? <div className="incident-episode-strip" aria-label="Camera episodes">{episodes.filter((episode) => !episode.part_index).map((episode) => <div key={episode.id}>
      <strong>{cameraName(episode.camera_id)}</strong><span>{date(episode.start_at)} — {date(episode.end_at)}</span>
      <small>{sceneCoverageText(episode.coverage)}</small>
      {onSelectEpisode ? <button type="button" disabled={!episode.clip} aria-pressed={activeEpisodeId === episode.id} onClick={() => onSelectEpisode(episode.id)}>Play {cameraName(episode.camera_id)} episode</button> : null}
      {!episode.clip ? <small>{episode.playback_note || "Episode recording reference unavailable."}</small> : null}
      {episode.camera_id && Number.isFinite(Date.parse(episode.start_at)) ? <a href={recordingsHref({ cameraId: episode.camera_id, epoch: Date.parse(episode.start_at) / 1000, source: "main", eventId: episode.clip?.representative_event_id })}>Open {cameraName(episode.camera_id)} recording timeline</a> : null}
    </div>)}</div> : null}
    {activeEpisodeId ? <div className="incident-scene-playback-controls"><span>Playing {cameraName(episodes.find((episode) => episode.id === activeEpisodeId)?.camera_id)}</span>{onNextEpisode ? <button type="button" onClick={onNextEpisode}>Next camera episode</button> : null}{onStopPlayback ? <button type="button" onClick={onStopPlayback}>Stop scene playback</button> : null}</div> : null}
    <div className="incident-scene-coverage" role="status"><p>{sceneCoverageText(scene?.coverage)}</p>
      {scene?.coverage?.analyzed_through ? <small>Analyzed through {date(scene.coverage.analyzed_through)}</small> : null}
      {scene?.coverage?.gaps?.length ? <ul>{scene.coverage.gaps.map((gap, index) => <li key={index}>{typeof gap === "string" ? gap : `${date(gap.start_at)} — ${date(gap.end_at)}${gap.reason ? ` · ${String(gap.reason).replaceAll("_", " ")}` : ""}`}</li>)}</ul> : null}
    </div>
    <details className="incident-scene-disclosure incident-scene-objects"><summary>Observed objects</summary>
    {scene?.continuity_uncertain ? <p className="incident-scene-continuity">Some sightings may show the same subject. The list does not establish a count of unique identities.</p> : null}
    <div className="incident-summary-objects">{objects.map((object) => <div className={`inspector-detection summary${String(selectedObjectId) === String(object.id) ? " selected" : ""}`} key={object.id}>
      <button type="button" className="incident-scene-object" aria-pressed={String(selectedObjectId) === String(object.id)} onClick={() => { setSelectedObjectId(object.id); setSelectedObservationId(null); setSelectedIds([]); setLabel(object.label); }}>
        <strong>{sceneObjectLabel(object)}</strong><span>{object.certainty === "uncertain" || object.certainty === "possible" ? "Uncertain observation" : "Observed"}</span>
        {object.continuity_uncertain ? <small className="incident-object-continuity">May be the same {object.label === "person" ? "person" : "object"} as another sighting</small> : null}
        <small>{date(object.first_seen_at)}{object.last_seen_at && object.last_seen_at !== object.first_seen_at ? ` — ${date(object.last_seen_at)}` : ""}</small>
      </button>
      {onFindSimilar && object.source_event_id && Number.isInteger(object.object_index) ? <button type="button" onClick={() => onFindSimilar({ eventId: object.source_event_id, objectIndex: object.object_index, label: object.label, trackId: object.track_id ?? null })}>Find similar</button> : null}
    </div>)}</div>
    {!objects.length ? <p>No object observations are available in the analyzed evidence.</p> : null}
    </details>
    <details className="incident-scene-disclosure incident-scene-activity" aria-label="Scene activity"><summary>Activity</summary>
      {activity.length ? <ol>{(allActivity ? activity : activity.slice(0, 20)).map((entry, index) => {
        const subject = objects.find((object) => String(object.id) === String(entry.object_id));
        const observation = subject?.observations?.find((item) => String(item.id) === String(entry.observation_id));
        return <li key={`${entry.kind}-${entry.observation_id}-${index}`}>
          <strong>{sceneActivityText({ ...entry, label: subject ? sceneObjectLabel(subject) : entry.label })}</strong><small>{cameraName(entry.camera_id)} · {date(entry.captured_at)}</small>
          {observation ? <button type="button" onClick={() => { setSelectedObjectId(subject.id); setSelectedObservationId(observation.id); setSelectedIds([]); setLabel(subject.label); onSelectObservation?.(observation); }}>View supporting observation</button> : null}
        </li>;
      })}</ol> : <p>No activity changes have been established from the available observations.</p>}
      {activity.length > 20 ? <button type="button" onClick={() => setAllActivity((current) => !current)}>{allActivity ? "Show fewer activity changes" : `Show all ${activity.length} activity changes`}</button> : null}
    </details>
    {selectedObject ? <section className="incident-scene-evidence" aria-label="Supporting observations"><h4>{sceneObjectLabel(selectedObject)}: supporting observations</h4>
      <div className="incident-observation-list">{observations.map((observation, index) => <button key={observation.id ?? index} type="button" aria-pressed={String(selectedObservationId) === String(observation.id ?? index)} onClick={() => { setSelectedObservationId(observation.id ?? index); onSelectObservation?.(observation); }}>{cameraName(observation.camera_id)} · {date(observation.captured_at)}</button>)}</div>
      {!observations.length ? <p>No supporting observations are available.</p> : null}
      {selectedObservation ? <><ObservationImage observation={selectedObservation} timeZone={timeZone} />
        {selectedObservation.camera_id && Number.isFinite(Date.parse(selectedObservation.captured_at)) ? <a href={recordingsHref({ cameraId: selectedObservation.camera_id, epoch: Date.parse(selectedObservation.captured_at) / 1000, source: "main", eventId: selectedObservation.event_id })}>Open recording at {date(selectedObservation.captured_at)}</a> : null}
        <details><summary>Detection details</summary><p>Detector confidence: {Math.round(Number(selectedObservation.confidence || 0) * 100)}%</p></details></> : null}
    </section> : null}
    <section className="incident-scene-alerts" aria-label="Notification policy"><h4>Notification policy</h4>
      <p>{sceneNotificationSummary(scene?.alert_decisions)}</p>
      <small>This describes the policy decision, not confirmation that a notification was delivered. The scene includes every retained object observation.</small>
      {scene?.alert_decisions?.length ? <details><summary>Policy decision details</summary>{scene.alert_decisions.map((decision, decisionIndex) => <div key={decision.event_id ?? decisionIndex}>
        <small>{date(scene.events?.find((event) => Number(event.id) === Number(decision.event_id))?.created_at)}</small>
        <ul>{(decision.objects || []).map((object, index) => <li key={`${object.label}-${index}`}><strong>{object.label}</strong>: {object.eligible ? "Meets notification criteria" : "Does not meet notification criteria"}
          {object.zones?.length ? <small> · {object.zones.join(", ")}</small> : null}
          {object.reasons?.length ? <details><summary>Technical policy reasons</summary><ul>{object.reasons.map((reason) => <li key={reason}>{String(reason).replaceAll("_", " ")}</li>)}</ul></details> : null}
        </li>)}</ul>
      </div>)}</details> : null}
    </section>
    {scene?.incident_id && scene.revision != null && Number.isFinite(Number(scene.revision)) ? <details className="incident-corrections"><summary>Correct this incident</summary><p>Changes preserve the original evidence. Incident ID: <code>{scene.incident_id}</code></p>
      <form onSubmit={correct}><label>Correction<select value={operation} onChange={(event) => { setOperation(event.target.value); setSelectedIds([]); setError(""); }}>
        <option value="label">Correct object label</option><option value="associate">Associate object observations</option><option value="separate">Separate object observations</option><option value="split">Separate camera episodes</option><option value="merge">Merge incidents</option>
      </select></label>
      {operation === "label" ? <label>Label for {selectedObject?.label || "selected object"}<input value={label} onChange={(event) => setLabel(event.target.value)} required disabled={!selectedObject} /></label>
        : operation === "merge" ? <label>Other incident IDs<input value={mergeIds} onChange={(event) => setMergeIds(event.target.value)} required placeholder="Comma separated incident IDs" /></label>
          : <fieldset><legend>{operation === "separate" ? "Select observations from the selected object" : operation === "associate" ? "Select observations of the same object" : "Select episodes for a separate incident"}</legend>{correctionItems.map((item) => <label key={item.id}><input type="checkbox" checked={selectedIds.includes(item.id)} onChange={(event) => setSelectedIds((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} />{item.label}</label>)}</fieldset>}
      <button type="submit" disabled={busy}>{busy ? "Saving…" : "Save correction"}</button>
      {error ? <p role="alert">{error}</p> : null}{notice ? <p role="status">{notice}</p> : null}
      </form></details> : null}
  </section>;
}
