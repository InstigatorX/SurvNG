import React, { useEffect, useRef, useState } from "react";
import { appUrl, fetch } from "../shared/api.js";

function boxStyle(box, width, height) {
  if (!width || !height || !box) return null;
  return {
    left: `${(box.x1 / width) * 100}%`,
    top: `${(box.y1 / height) * 100}%`,
    width: `${((box.x2 - box.x1) / width) * 100}%`,
    height: `${((box.y2 - box.y1) / height) * 100}%`,
  };
}

async function readError(response, fallback) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
  } catch {
    // The response body is not JSON.
  }
  return fallback;
}

export function FaceFindingPanel({ timeZone = "UTC" }) {
  const [samples, setSamples] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [faces, setFaces] = useState([]);
  const [falseBoxes, setFalseBoxes] = useState([]);
  const [draft, setDraft] = useState(null);
  const [dropReason, setDropReason] = useState("");
  const [candidatePath, setCandidatePath] = useState("");
  const [baselinePath, setBaselinePath] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const stageRef = useRef(null);
  const drag = useRef(null);
  const selected = samples.find((sample) => sample.id === selectedId) || samples[0] || null;

  async function loadQueue() {
    const response = await fetch("/api/face-detection/samples?limit=20");
    if (!response.ok) throw new Error(await readError(response, "Unable to load face-finding crops"));
    const payload = await response.json();
    setSamples(payload.samples || []);
    setSelectedId((current) => current || payload.samples?.[0]?.id || null);
  }

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/face-detection/samples?limit=20", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await readError(response, "Unable to load face-finding crops"));
        return response.json();
      })
      .then((payload) => {
        setSamples(payload.samples || []);
        setSelectedId(payload.samples?.[0]?.id || null);
      })
      .catch((loadError) => {
        if (loadError.name !== "AbortError") setError(loadError.message || "Unable to load face-finding crops");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    setFaces([]);
    setFalseBoxes([]);
    setDraft(null);
    setDropReason("");
  }, [selected?.id]);

  async function run(action) {
    setBusy(true);
    setError("");
    try {
      const result = await action();
      if (typeof result === "string") setMessage(result);
      await loadQueue();
    } catch (actionError) {
      setError(actionError.message || "Face finding request failed");
    } finally {
      setBusy(false);
    }
  }

  function pointFromEvent(event) {
    const image = stageRef.current?.querySelector("img");
    if (!image?.naturalWidth || !image.naturalHeight) return null;
    const rect = image.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: Math.min(image.naturalWidth, Math.max(0, ((event.clientX - rect.left) / rect.width) * image.naturalWidth)),
      y: Math.min(image.naturalHeight, Math.max(0, ((event.clientY - rect.top) / rect.height) * image.naturalHeight)),
    };
  }

  function beginDraw(event) {
    if (event.button !== 0) return;
    const point = pointFromEvent(event);
    if (!point) return;
    drag.current = point;
    setDraft({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  }

  function moveDraw(event) {
    if (!drag.current) return;
    const point = pointFromEvent(event);
    if (!point) return;
    setDraft({
      x1: Math.min(drag.current.x, point.x),
      y1: Math.min(drag.current.y, point.y),
      x2: Math.max(drag.current.x, point.x),
      y2: Math.max(drag.current.y, point.y),
    });
  }

  function finishDraw(event) {
    const start = drag.current;
    drag.current = null;
    setDraft(null);
    if (!start) return;
    const point = pointFromEvent(event);
    if (!point) return;
    const box = {
      x1: Math.min(start.x, point.x),
      y1: Math.min(start.y, point.y),
      x2: Math.max(start.x, point.x),
      y2: Math.max(start.y, point.y),
    };
    if (box.x2 - box.x1 < 4 || box.y2 - box.y1 < 4) return;
    setFaces((current) => [...current, box]);
  }

  async function saveLabel(body) {
    if (!selected) return;
    await run(async () => {
      const response = await fetch(`/api/face-detection/samples/${selected.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await readError(response, "Unable to save the detection label"));
      return "Detection label saved. It is not a person name.";
    });
  }

  return (
    <div className="face-finding">
      <div className="face-finding-queue">
        <div className="queue-toolbar-copy">
          <strong>Face finding</strong>
          <small>Label the upper-body crop. This does not enroll or name anyone.</small>
        </div>
        <ul>
          {samples.map((sample) => (
            <li key={sample.id}>
              <button type="button" className={sample.id === selected?.id ? "active" : ""} onClick={() => setSelectedId(sample.id)}>
                <strong>{sample.camera_id}</strong>
                <small>{sample.calendar_day} · {sample.baseline_miss || !sample.baseline_ran ? "no baseline face" : "baseline face"}</small>
              </button>
            </li>
          ))}
        </ul>
        {!samples.length ? <p className="empty-state">No unlabeled upper-body crops yet.</p> : null}
      </div>
      <div className="face-finding-work">
        {error ? <div className="face-load-error" role="alert">{error}</div> : null}
        {message ? <div className="save-status" role="status">{message}</div> : null}
        {selected ? (
          <div
            className="face-finding-stage"
            ref={stageRef}
            onPointerDown={beginDraw}
            onPointerMove={moveDraw}
            onPointerUp={finishDraw}
            onPointerLeave={() => { drag.current = null; setDraft(null); }}
          >
            <img src={appUrl(selected.crop_url)} alt={`Upper-body crop from ${selected.camera_id}`} draggable="false" />
            {(selected.baseline_predictions || []).map((prediction, index) => (
              <button
                key={`baseline-${index}`}
                type="button"
                className="face-finding-box baseline"
                style={boxStyle(prediction.box, selected.crop_width, selected.crop_height)}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation();
                  setFalseBoxes((current) => [...current, prediction.box]);
                }}
                title="Mark this baseline box as not a face"
              />
            ))}
            {faces.map((box, index) => <span key={`face-${index}`} className="face-finding-box face" style={boxStyle(box, selected.crop_width, selected.crop_height)} />)}
            {falseBoxes.map((box, index) => <span key={`false-${index}`} className="face-finding-box false" style={boxStyle(box, selected.crop_width, selected.crop_height)} />)}
            {draft ? <span className="face-finding-box face" style={boxStyle(draft, selected.crop_width, selected.crop_height)} /> : null}
          </div>
        ) : <div className="empty-state">Materialize crops from recorded person observations.</div>}
        <div className="face-finding-actions">
          <button type="button" disabled={busy || !selected || (!faces.length && !falseBoxes.length)} onClick={() => saveLabel({
            annotations: [
              ...faces.map((box) => ({ kind: "face", box })),
              ...falseBoxes.map((box) => ({ kind: "not_a_face", box })),
            ],
          })}>Save boxes</button>
          <button type="button" className="subtle" disabled={busy || !selected} onClick={() => saveLabel({ annotations: [{ kind: "no_face" }] })}>No face</button>
          <button type="button" className="subtle" disabled={!faces.length && !falseBoxes.length} onClick={() => { setFaces([]); setFalseBoxes([]); }}>Clear boxes</button>
          <input value={dropReason} onChange={(event) => setDropReason(event.target.value)} placeholder="Drop reason" aria-label="Drop reason" />
          <button type="button" className="subtle" disabled={busy || !selected || !dropReason.trim()} onClick={() => saveLabel({ annotations: [{ kind: "drop" }], drop_reason: dropReason.trim() })}>Drop</button>
          <button type="button" className="subtle" disabled={busy} onClick={() => run(async () => {
            const response = await fetch("/api/face-detection/samples/materialize", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ timezone: timeZone || "UTC", limit: 50 }),
            });
            if (!response.ok) throw new Error(await readError(response, "Unable to materialize crops"));
            const payload = await response.json();
            return `Stored ${payload.stored || 0} upper-body crops.`;
          })}>Materialize</button>
          <button type="button" className="subtle" disabled={busy} onClick={() => run(async () => {
            const response = await fetch("/api/face-detection/splits", { method: "POST" });
            if (!response.ok) throw new Error(await readError(response, "Unable to freeze day splits"));
            return "Day splits frozen.";
          })}>Freeze splits</button>
          <button type="button" className="subtle" disabled={busy} onClick={() => run(async () => {
            const response = await fetch("/api/face-detection/export", { method: "POST" });
            if (!response.ok) throw new Error(await readError(response, "Unable to export detection crops"));
            const payload = await response.json();
            return `Exported ${payload.samples || 0} crops to ${payload.directory}.`;
          })}>Export</button>
        </div>
        <form className="face-finding-score" onSubmit={(event) => {
          event.preventDefault();
          void run(async () => {
            const response = await fetch("/api/face-detection/score", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                candidate_model_path: candidatePath,
                baseline_model_path: baselinePath,
              }),
            });
            if (!response.ok) throw new Error(await readError(response, "Unable to score the face detector"));
            const payload = await response.json();
            const reasons = payload.gate?.reasons?.length ? payload.gate.reasons.join(" ") : "Promotion gate passed.";
            return `${reasons} Embedding fingerprint ${payload.embedding_fingerprint || "is unset"}. ${payload.activation || ""}`;
          });
        }}>
          <input value={candidatePath} onChange={(event) => setCandidatePath(event.target.value)} placeholder="Candidate detector .xml" aria-label="Candidate face detector model" />
          <input value={baselinePath} onChange={(event) => setBaselinePath(event.target.value)} placeholder="Baseline detector .xml" aria-label="Baseline face detector model" />
          <button type="submit" disabled={busy || !candidatePath || !baselinePath}>Score IR</button>
        </form>
      </div>
    </div>
  );
}
