import React, { useEffect, useRef, useState } from "react";
import { appUrl, fetch } from "../shared/api.js";
import "./DetectionReviews.css";

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Review request failed. Refresh the evidence and try again.");
  return body;
}
const endpoint = (source, id) => source === "event" ? `/api/events/${id}` : `/api/motion-audits/${id}`;

export function DetectionReviewControl({ source = "event", sourceId, labels = [], initialObservation = "" }) {
  const [open, setOpen] = useState(false);
  if (!Number.isInteger(Number(sourceId)) || Number(sourceId) <= 0) return null;
  return <details className="detection-review" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Review detections for training</summary>
    {open ? <ReviewEditor key={`${source}:${sourceId}:${initialObservation}`} source={source} sourceId={sourceId} labels={labels} initialObservation={initialObservation} /> : null}
  </details>;
}

export function ReviewEditor({ source, sourceId, labels = [], initialObservation = "" }) {
  const [evidence, setEvidence] = useState(null);
  const [observation, setObservation] = useState(initialObservation);
  const [generation, setGeneration] = useState(0);
  const [reviews, setReviews] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [imageReady, setImageReady] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [selected, setSelected] = useState("");
  const [box, setBox] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [verdict, setVerdict] = useState("confirmed");
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [scope, setScope] = useState("camera");
  const [location, setLocation] = useState(true);
  const [replace, setReplace] = useState(false);
  const [saved, setSaved] = useState(null);
  const drag = useRef(null);
  const retry = useRef(null);
  const labelListId = React.useId();
  useEffect(() => {
    const controller = new AbortController();
    setEvidence(null); setError(""); setImageReady(false); setSelected(""); setBox(null); setReplace(false); setSaved(null);
    Promise.all([
      request(`${endpoint(source, sourceId)}/review-evidence?observation_id=${encodeURIComponent(observation)}`, { signal: controller.signal }),
      request(`/api/detection-corrections?source=${source}&source_id=${sourceId}&status=active&limit=100`, { signal: controller.signal }),
    ]).then(([data, list]) => { setEvidence(data); setReviews(list.items); }).catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [source, sourceId, observation, generation]);
  const active = reviews.filter((r) => r.asset.image_hash === evidence?.image_hash && r.status === "active");
  const chosen = evidence?.objects.find((item) => item.key === selected);
  const isNegative = verdict === "verified_negative";
  const conflicts = active.filter((r) => r.annotation_key === (isNegative ? "negative" : (chosen?.annotation_key || chosen?.observation_id || selected))
    || (r.verdict === "verified_negative" && ["confirmed", "wrong_label", "missed"].includes(verdict))
    || (isNegative && ["confirmed", "wrong_label", "missed"].includes(r.verdict)));
  function choose(item) {
    setSelected(item.key); setBox(item.box); setDrawing(false); setReplace(false);
    setVerdict(item.incident_eligible ? "confirmed" : "missed"); setLabel(item.label);
  }
  function point(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
  }
  function move(event) {
    if (!drag.current) return;
    const [x,y] = point(event), [sx,sy] = drag.current;
    setBox([Math.min(x,sx),Math.min(y,sy),Math.max(x,sx),Math.max(y,sy)]);
  }
  async function save() {
    setBusy(true); setError("");
    const values = { evidence_revision:evidence.revision, observation_id:observation,
      object_key:selected, source:chosen ? (chosen.incident_eligible ? "detection" : "hidden_candidate") : "drawn",
      verdict, corrected_label:label.trim() || null, box, scope, location_rule:location, note,
      supersedes:replace ? Object.fromEntries(conflicts.map((r) => [String(r.id),r.revision])) : {},
    };
    const signature = JSON.stringify(values);
    if (retry.current?.signature !== signature) retry.current = { signature, key:crypto.randomUUID() };
    try {
      const row = await request(`${endpoint(source,sourceId)}/${isNegative ? "review" : "corrections"}`, {
        method:isNegative ? "PUT" : "POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({...values,idempotency_key:retry.current.key}),
      });
      setSaved(row); setReplace(false);
      const list = await request(`/api/detection-corrections?source=${source}&source_id=${sourceId}&status=active&limit=100`);
      setReviews(list.items);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  async function revoke(row) {
    setBusy(true); setError("");
    try {
      await request(`/api/detection-corrections/${row.id}?revision=${row.revision}`,{method:"DELETE"});
      setReviews((previous) => previous.filter((r) => r.id !== row.id)); setSaved(null);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  const needsLabel = verdict === "wrong_label" || (!chosen && !isNegative);
  const validBox = box && box[2] > box[0] && box[3] > box[1];
  return <div>
    <p>Save reviewed examples for training. Future detection matching is off. Existing incident decisions stay unchanged.</p>
    {error ? <p role="alert">{error} <button type="button" onClick={() => setGeneration((n) => n+1)}>Reload evidence</button></p> : null}
    {!evidence && !error ? <p role="status">Loading review image…</p> : null}
    {evidence ? <>
      {evidence.supporting_observations.length ? <label>Evidence image<select value={observation} onChange={(e) => setObservation(e.target.value)} disabled={busy}>
        <option value="">Representative image</option>
        {evidence.supporting_observations.map((id, index) => <option key={id} value={id}>Observation {index+1}</option>)}
      </select></label> : null}
      <label><input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} /> Show hidden candidates</label>
      <div className="review-image">
        <img alt="Exact image being reviewed" src={appUrl(`/api/detection-corrections/source-image?${new URLSearchParams({source,source_id:sourceId,revision:evidence.revision,observation_id:observation})}`)} onLoad={() => setImageReady(true)} onError={() => { setImageReady(false); setError("Image unavailable or changed. Reload before saving."); }} />
        {imageReady ? <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Draw annotation on image" style={{touchAction:drawing ? "none" : "auto"}}
          onPointerDown={(event) => { if (!drawing || !evidence.can_review || busy) return; event.currentTarget.setPointerCapture(event.pointerId); drag.current=point(event); setSelected(""); setBox(null); }}
          onPointerMove={move} onPointerUp={(event) => { move(event); drag.current=null; }} onPointerCancel={() => { drag.current=null; setBox(null); }}>
          {evidence.objects.filter((item) => item.incident_eligible || hidden).map((item) => <rect key={item.key} x={item.box[0]} y={item.box[1]} width={item.box[2]-item.box[0]} height={item.box[3]-item.box[1]} className={item.incident_eligible ? "candidate" : "candidate hidden-candidate"} onClick={() => { if (!drawing && !busy) choose(item); }}><title>{item.label}</title></rect>)}
          {box ? <rect className="selected-box" x={box[0]} y={box[1]} width={box[2]-box[0]} height={box[3]-box[1]} /> : null}
        </svg> : null}
      </div>
      <div className="review-object-list">{evidence.objects.filter((item) => item.incident_eligible || hidden).map((item,index) => <button key={item.key} type="button" onClick={() => choose(item)} disabled={busy}>{item.label} {index+1}{item.incident_eligible ? "" : " (hidden)"}</button>)}</div>
      {evidence.can_review ? <fieldset disabled={busy}>
        <legend>Review this image</legend>
        <button type="button" onClick={() => { setDrawing(true); setSelected(""); setBox(null); setVerdict("missed"); setLabel(""); setReplace(false); }}>Draw a missed object</button>
        {drawing ? <p>Drag a rectangle around the missed object, then enter its label.</p> : null}
        {source === "motion_audit" ? <button type="button" onClick={() => { setVerdict("verified_negative"); setDrawing(false); setSelected(""); setBox(null); setReplace(false); }}>Nothing of interest in this image</button> : null}
        {chosen ? <label>Verdict<select value={verdict} onChange={(e) => { setVerdict(e.target.value); setReplace(false); }}>
          <option value="confirmed">Correct: {chosen.label}</option><option value="not_object">Not a {chosen.label}</option><option value="wrong_label">It is actually…</option>
          {!chosen.incident_eligible ? <option value="missed">Missed object: {chosen.label}</option> : null}
        </select></label> : null}
        {isNegative ? <p>This explicitly verifies the whole image as negative for training.</p> : null}
        {needsLabel ? <label>Correct label<input list={labelListId} value={label} maxLength={100} onChange={(e) => setLabel(e.target.value)} /><datalist id={labelListId}>{[...new Set([...labels,...evidence.objects.map((o) => o.label)])].map((value) => <option key={value} value={value} />)}</datalist></label> : null}
        {!isNegative ? <><label>Future scope<select value={scope} onChange={(e) => setScope(e.target.value)}><option value="camera">This camera</option><option value="global">All cameras</option></select></label>
          <label><input type="checkbox" checked={location} onChange={(e) => setLocation(e.target.checked)} /> Only at this spot on this camera</label></> : null}
        <label>Note<input value={note} maxLength={2000} onChange={(e) => setNote(e.target.value)} /></label>
        {conflicts.length ? <label><input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} /> Replace conflicting reviews {conflicts.map((r) => `#${r.id}`).join(", ")}</label> : null}
        <button type="button" onClick={save} disabled={!imageReady || (!isNegative && !validBox) || (needsLabel && !label.trim()) || (conflicts.length > 0 && !replace)}>{busy ? "Saving…" : "Save review"}</button>
      </fieldset> : <p>Administrator access is required to save reviews.</p>}
      {saved ? <p role="status">Saved review #{saved.id}. <button type="button" disabled={busy} onClick={() => revoke(saved)}>Undo review</button></p> : null}
      {active.length ? <ul>{active.map((row) => <li key={row.id}>#{row.id}: {row.verdict.replaceAll("_"," ")} {row.label || ""} {evidence.can_review ? <button type="button" disabled={busy} onClick={() => revoke(row)}>Revoke</button> : null}</li>)}</ul> : <p>No active reviews on this image.</p>}
    </> : null}
  </div>;
}

export function DetectionReviewsAdmin() {
  const [items,setItems] = useState([]), [queue,setQueue] = useState([]);
  const [next,setNext] = useState(null), [queueNext,setQueueNext] = useState(null);
  const [status,setStatus] = useState("active"), [camera,setCamera] = useState(""), [verdict,setVerdict] = useState("");
  const [error,setError] = useState(""), [busy,setBusy] = useState(false), [audit,setAudit] = useState(null);
  const [reload,setReload] = useState(0);
  useEffect(() => {
    const controller=new AbortController(); setBusy(true); setError("");
    Promise.all([request(`/api/detection-corrections?${new URLSearchParams({status,camera_id:camera,verdict})}`,{signal:controller.signal}),request(`/api/detection-corrections/review-queue?${new URLSearchParams({camera_id:camera})}`,{signal:controller.signal})])
      .then(([list,pending]) => {setItems(list.items);setNext(list.next_cursor);setQueue(pending.items);setQueueNext(pending.next_cursor);})
      .catch((e) => {if(!controller.signal.aborted)setError(e.message);}).finally(() => {if(!controller.signal.aborted)setBusy(false);});
    return () => controller.abort();
  },[status,camera,verdict,reload]);
  async function more(isQueue) {
    setBusy(true);setError("");
    try {
      const result=await request(isQueue ? `/api/detection-corrections/review-queue?${new URLSearchParams({camera_id:camera,cursor:queueNext})}` : `/api/detection-corrections?${new URLSearchParams({status,camera_id:camera,verdict,cursor:next})}`);
      if(isQueue){setQueue((old)=>[...old,...result.items]);setQueueNext(result.next_cursor);} else {setItems((old)=>[...old,...result.items]);setNext(result.next_cursor);}
    }catch(e){setError(e.message);}finally{setBusy(false);}
  }
  async function revoke(row) {
    setBusy(true);setError("");
    try {await request(`/api/detection-corrections/${row.id}?revision=${row.revision}`,{method:"DELETE"});setReload((n)=>n+1);}catch(e){setError(e.message);setBusy(false);}
  }
  return <section className="detection-settings-card wide-card detection-review">
    <h3>Detection reviews</h3><p>Reviewed training examples. Appearance matching is off; match counts remain zero.</p>
    <label>Camera ID<input value={camera} onChange={(e)=>setCamera(e.target.value)} /></label>
    <label>Status<select value={status} onChange={(e)=>setStatus(e.target.value)}>{["active","revoked","superseded",""].map((v)=><option key={v} value={v}>{v || "All"}</option>)}</select></label>
    <label>Verdict<select value={verdict} onChange={(e)=>setVerdict(e.target.value)}>{["","not_object","wrong_label","confirmed","missed","verified_negative"].map((v)=><option key={v} value={v}>{v.replaceAll("_"," ") || "All"}</option>)}</select></label>
    <button type="button" onClick={()=>setReload((n)=>n+1)} disabled={busy}>Refresh</button>
    {error ? <p role="alert">{error}</p> : null}{busy ? <p role="status">Loading…</p> : null}
    {!busy && !items.length ? <p>No matching reviews.</p> : null}
    <ul className="review-admin-list">{items.map((row)=><li key={row.id}><a href={appUrl(`/api/detection-corrections/assets/${row.asset_id}`)} target="_blank" rel="noreferrer"><img alt={`Review ${row.id}`} src={appUrl(`/api/detection-corrections/assets/${row.asset_id}?review_id=${row.id}`)} loading="lazy" /></a><span>#{row.id} · {row.asset.camera_id} · {row.verdict.replaceAll("_"," ")} {row.original_label ? `(${row.original_label})` : ""} {row.label || ""}<br />{row.status} · matches: 0 · last matched: never<br />{row.created_by} · {new Date(row.created_at * 1000).toLocaleString()}<br />{row.note}</span>{row.status === "active" ? <button type="button" disabled={busy} onClick={()=>revoke(row)}>Revoke</button> : null}</li>)}</ul>
    {next ? <button type="button" disabled={busy} onClick={()=>more(false)}>More reviews</button> : null}
    <h4>Motion images awaiting review</h4><p>Probable missed detections appear first. Refresh after reviewing.</p>
    {!busy && !queue.length ? <p>No pending images in this page.</p> : null}
    <ul>{queue.map((row)=><li key={row.id}><button type="button" onClick={()=>setAudit(row.id)}>{row.camera_id} · {row.created_at}{row.probable_miss ? " · probable miss" : ""}</button></li>)}</ul>
    {queueNext ? <button type="button" disabled={busy} onClick={()=>more(true)}>More images</button> : null}
    {audit ? <div><button type="button" onClick={()=>setAudit(null)}>Close image review</button><ReviewEditor key={audit} source="motion_audit" sourceId={audit} /></div> : null}
  </section>;
}
