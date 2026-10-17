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

function activityText(activity) {
  if (!activity) return "";
  const counts = activity.total > 0 ? ` ${activity.current} of ${activity.total}` : "";
  const detail = activity.detail ? ` · ${activity.detail}` : "";
  return `${activity.phase || activity.label}${counts}${detail}`;
}

function ActionButton({ idle, pending, running, className = "", disabled = false, onClick, type = "button" }) {
  return (
    <button
      type={type}
      className={[className, running ? "is-running" : ""].filter(Boolean).join(" ")}
      disabled={disabled}
      aria-busy={running || undefined}
      onClick={onClick}
    >
      {running ? pending : idle}
    </button>
  );
}

function formatCropWhen(epoch, timeZone) {
  const date = new Date(Number(epoch) * 1000);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  const options = { hour: "numeric", minute: "2-digit" };
  try {
    return new Intl.DateTimeFormat("en-US", timeZone ? { ...options, timeZone } : options).format(date);
  } catch {
    return new Intl.DateTimeFormat("en-US", options).format(date);
  }
}

function formatCropDay(day) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(day || ""));
  if (!match) return day || "Unknown day";
  return new Intl.DateTimeFormat("en-US", { month: "long", day: "numeric", year: "numeric" }).format(
    new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3])),
  );
}

function detectorLabel(sample) {
  const count = Array.isArray(sample.baseline_predictions) ? sample.baseline_predictions.length : 0;
  if (sample.baseline_ran && !sample.baseline_miss && count > 0) {
    return count === 1 ? "Detector marked a face" : `Detector marked ${count} faces`;
  }
  if (!sample.baseline_ran) return "Detector did not run";
  return "Detector marked nothing";
}

function groupSamples(samples) {
  const groups = new Map();
  for (const sample of samples) {
    const day = sample.calendar_day || "Unknown day";
    if (!groups.has(day)) groups.set(day, []);
    groups.get(day).push(sample);
  }
  return [...groups.entries()]
    .sort((left, right) => right[0].localeCompare(left[0]))
    .map(([day, items]) => ({
      day,
      samples: items.slice().sort((left, right) => Number(right.captured_epoch) - Number(left.captured_epoch)),
    }));
}

function FaceFindingStatus({ busy, activity, message, error }) {
  const text = activityText(activity);
  const percent = activity?.total > 0 && activity.current > 0
    ? Math.max(0, Math.min(100, Math.round((activity.current / activity.total) * 100)))
    : null;
  return (
    <div className="face-finding-status" aria-live="polite">
      {busy ? (
        <>
          <div className="face-finding-status-line">
            <span className="face-finding-spinner" aria-hidden="true" />
            <span>{text}</span>
          </div>
          <div
            className="face-finding-progress-track"
            role="progressbar"
            aria-label={text}
            aria-valuemin={0}
            aria-valuemax={100}
            {...(percent == null ? {} : { "aria-valuenow": percent })}
          >
            <div
              className={percent == null ? "face-finding-progress-bar" : "face-finding-progress-bar is-determinate"}
              style={percent == null ? undefined : { width: `${Math.max(percent, 4)}%` }}
            />
          </div>
        </>
      ) : null}
      {!busy && message ? <div className="save-status" role="status">{message}</div> : null}
      {error ? <div className="face-load-error" role="alert">{error}</div> : null}
    </div>
  );
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
  const [activity, setActivity] = useState(null);
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.matchMedia("(max-width: 760px)").matches);
  const [drawMode, setDrawMode] = useState(false);
  const stageRef = useRef(null);
  const drag = useRef(null);
  const drawing = !narrow || drawMode;
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

  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const sync = () => {
      setNarrow(media.matches);
      if (!media.matches) setDrawMode(false);
    };
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  async function run(label, action) {
    setBusy(true);
    setActivity({ label, phase: label, current: 0, total: 0, detail: "" });
    setError("");
    setMessage("");
    let stopped = false;
    const poll = (async () => {
      while (!stopped) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        if (stopped) return;
        try {
          const response = await fetch("/api/face-detection/activity");
          if (!response.ok || stopped) continue;
          const payload = await response.json();
          if (!payload?.active || stopped) continue;
          setActivity((current) => ({
            label: current?.label || label,
            phase: payload.phase || label,
            current: Number(payload.current || 0),
            total: Number(payload.total || 0),
            detail: payload.detail || "",
          }));
        } catch {
          // Keep the label from the button press if the progress read fails.
        }
      }
    })();
    try {
      const result = await action();
      if (typeof result === "string") setMessage(result);
      await loadQueue();
    } catch (actionError) {
      setError(actionError.message || "Face finding request failed");
    } finally {
      stopped = true;
      await poll;
      setBusy(false);
      setActivity(null);
    }
  }

  function working(label) {
    return busy && activity?.label === label;
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
    if (!drawing || event.button !== 0) return;
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

  async function saveLabel(body, label) {
    if (!selected) return;
    await run(label, async () => {
      const response = await fetch(`/api/face-detection/samples/${selected.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await readError(response, "Unable to save the detection label"));
      return "Detection label saved. It is not a person name.";
    });
  }

  const groups = groupSamples(samples);
  return (
    <div className="face-finding">
      <ol className="face-finding-steps">
        <li className={samples.length ? "" : "is-current"}>
          <strong>1. Load crops</strong>
          <span>Copy recent recorded people into the list. Newer days appear only after you load again.</span>
        </li>
        <li className={samples.length ? "is-current" : ""}>
          <strong>2. Mark each crop</strong>
          <span>Draw a box around every face, or choose No face. Drop a crop that should not train the detector.</span>
        </li>
        <li>
          <strong>3. Export after three days</strong>
          <span>Freeze those days, export, train outside SurvNG, then score the new model.</span>
        </li>
      </ol>
      <div className="face-finding-queue">
        <div className="queue-toolbar-copy">
          <strong>Unlabeled crops</strong>
          <small>{samples.length ? `${samples.length} waiting. These labels do not name anyone.` : "None loaded yet."}</small>
          <ActionButton idle="Load crops" pending="Loading…" running={working("Loading crops")} disabled={busy} onClick={() => run("Loading crops", async () => {
            const response = await fetch("/api/face-detection/samples/materialize", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ timezone: timeZone || "UTC", limit: 50 }),
            });
            if (!response.ok) throw new Error(await readError(response, "Unable to load crops"));
            const payload = await response.json();
            const stored = Number(payload.stored || 0);
            if (!stored) {
              return `Loaded 0 new crops. ${payload.already || 0} already saved, ${payload.skipped || 0} had no recording, ${payload.undecoded || 0} could not be decoded.`;
            }
            return `Loaded ${stored} upper-body crops.`;
          })} />
        </div>
        {groups.map((group) => (
          <section key={group.day} className="face-finding-day">
            <h3>{formatCropDay(group.day)} <span>{group.samples.length}</span></h3>
            <ul>
              {group.samples.map((sample) => {
                const missed = sample.baseline_miss || !sample.baseline_ran;
                return (
                  <li key={sample.id}>
                    <button type="button" className={sample.id === selected?.id ? "face-finding-row active" : "face-finding-row"} onClick={() => setSelectedId(sample.id)}>
                      <img src={appUrl(sample.crop_url)} alt="" />
                      <span>
                        <strong>{formatCropWhen(sample.captured_epoch, timeZone)}</strong>
                        <small>{sample.camera_id}</small>
                        <small><i className={missed ? "face-finding-mark miss" : "face-finding-mark hit"} aria-hidden="true" />{detectorLabel(sample)}</small>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
        {!samples.length ? <p className="empty-state">Choose Load crops. This list does not fill itself.</p> : null}
      </div>
      <div className="face-finding-work" aria-busy={busy || undefined}>
        {selected ? <p className="face-finding-hint">{narrow ? "Choose Draw face, then drag a box around every face in this crop." : "Drag a box around every face. Click an orange box when the current detector marked something that is not a face."}</p> : null}
        {selected ? <p className="face-finding-key"><span><i className="face-finding-swatch face" />Your face box</span><span><i className="face-finding-swatch false" />Not a face</span><span><i className="face-finding-swatch baseline" />Current detector</span></p> : null}
        {narrow ? <button type="button" className="face-finding-draw-toggle" aria-pressed={drawMode} onClick={() => setDrawMode((current) => !current)}>{drawMode ? "Done drawing" : "Draw face"}</button> : null}
        {selected ? (
          <div
            className={drawing ? "face-finding-stage" : "face-finding-stage can-scroll"}
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
        ) : <div className="empty-state">Load crops, then mark the faces in each one.</div>}
        <FaceFindingStatus busy={busy} activity={activity} message={message} error={error} />
        <div className="face-finding-actions">
          <ActionButton idle="Save boxes" pending="Saving…" running={working("Saving boxes")} disabled={busy || !selected || (!faces.length && !falseBoxes.length)} onClick={() => saveLabel({
            annotations: [
              ...faces.map((box) => ({ kind: "face", box })),
              ...falseBoxes.map((box) => ({ kind: "not_a_face", box })),
            ],
          }, "Saving boxes")} />
          <ActionButton className="subtle" idle="No face" pending="Saving…" running={working("Saving the no-face label")} disabled={busy || !selected} onClick={() => saveLabel({ annotations: [{ kind: "no_face" }] }, "Saving the no-face label")} />
          <button type="button" className="subtle" disabled={busy || (!faces.length && !falseBoxes.length)} onClick={() => { setFaces([]); setFalseBoxes([]); setError(""); setMessage("Cleared the boxes on this crop."); }}>Clear boxes</button>
          <input value={dropReason} onChange={(event) => setDropReason(event.target.value)} placeholder="Drop reason" aria-label="Drop reason" disabled={busy} />
          <ActionButton className="subtle" idle="Drop" pending="Dropping…" running={working("Dropping this crop")} disabled={busy || !selected || !dropReason.trim()} onClick={() => saveLabel({ annotations: [{ kind: "drop" }], drop_reason: dropReason.trim() }, "Dropping this crop")} />
        </div>
        <details className="face-finding-later">
          <summary>After at least three labeled days</summary>
          <p>Freeze the days so the same day stays in one split. Export writes the crops for training outside SurvNG. Score a finished OpenVINO model before replacing the one that is running.</p>
          <div className="face-finding-actions">
            <ActionButton className="subtle" idle="Freeze days" pending="Freezing…" running={working("Freezing days")} disabled={busy} onClick={() => run("Freezing days", async () => {
              const response = await fetch("/api/face-detection/splits", { method: "POST" });
              if (!response.ok) throw new Error(await readError(response, "Unable to freeze day splits"));
              return "Days frozen into train, calibration, and held-out splits.";
            })} />
            <ActionButton className="subtle" idle="Export" pending="Exporting…" running={working("Exporting crops")} disabled={busy} onClick={() => run("Exporting crops", async () => {
              const response = await fetch("/api/face-detection/export", { method: "POST" });
              if (!response.ok) throw new Error(await readError(response, "Unable to export detection crops"));
              const payload = await response.json();
              return `Exported ${payload.samples || 0} crops to ${payload.directory}.`;
            })} />
          </div>
          <form className="face-finding-score" onSubmit={(event) => {
            event.preventDefault();
            void run("Scoring models", async () => {
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
            <input value={candidatePath} onChange={(event) => setCandidatePath(event.target.value)} placeholder="Candidate detector .xml" aria-label="Candidate face detector model" disabled={busy} />
            <input value={baselinePath} onChange={(event) => setBaselinePath(event.target.value)} placeholder="Baseline detector .xml" aria-label="Baseline face detector model" disabled={busy} />
            <ActionButton type="submit" idle="Score model" pending="Scoring…" running={working("Scoring models")} disabled={busy || !candidatePath || !baselinePath} />
          </form>
        </details>
      </div>
    </div>
  );
}
