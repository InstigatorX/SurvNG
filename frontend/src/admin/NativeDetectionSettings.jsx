import NativeBudgetSettings from "./NativeBudgetSettings.jsx";
import React, { useEffect, useState } from "react";
import { DEFAULT_STATIONARY_LABELS, NATIVE_DETECTION_FIELDS, detectionFieldValue, nativeDetectionError } from "../nativeDetectionSettings.mjs";

export function NativeDetectionSettings({ detector, updateConfig, modelClasses = [] }) {
  const [labelsText, setLabelsText] = useState(() => (detector.native?.stationary?.labels ?? DEFAULT_STATIONARY_LABELS).join(", "));
  const configuredLabels = JSON.stringify(detector.native?.stationary?.labels ?? DEFAULT_STATIONARY_LABELS);
  useEffect(() => {
    setLabelsText((current) => {
      const parsed = [...new Set(current.split(",").map((label) => label.trim().toLowerCase()).filter(Boolean))];
      return JSON.stringify(parsed) === configuredLabels ? current : JSON.parse(configuredLabels).join(", ");
    });
  }, [configuredLabels]);
  const [extraClass, setExtraClass] = useState("");
  const [addedClasses, setAddedClasses] = useState([]);
  const classes = [...new Set([...modelClasses, ...(detector.labels || []), ...(detector.native?.tracking_classes || []), ...addedClasses,
    ...Object.keys(detector.event_class_confirmation_frames || {}), ...Object.keys(detector.event_class_confidence_thresholds || {})])].sort();
  const trackedClasses = detector.native?.tracking_classes ?? null;
  const update = (path, value) => updateConfig(["detector", ...path.split(".")], value);
  const numericFields = (group) => <div className="form-grid">{NATIVE_DETECTION_FIELDS.filter((field) => field.group === group).map((field) => <label key={field.path}>{field.label}<input type="number" aria-label={field.label} min={field.min} max={field.max} step={field.step} value={detectionFieldValue(detector, field)} onChange={(event) => update(field.path, event.target.value === "" ? "" : Number(event.target.value))} />{field.help ? <small>{field.help}</small> : null}</label>)}</div>;
  function override(key, label, value) {
    const next = { ...detector[key] };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    update(key, next);
  }
  const error = nativeDetectionError(detector);
  return <section className="sub-panel detection-settings-card">
    <h3>Native detection and tracking</h3>
    <p>Live/substream → gvadetect → gvatrack → zone checks → incident policy. Tracking uses short-term-imageless; the inference interval controls how often sampled frames run detection. Save settings to apply changes.</p>
    <div className="form-grid">
      <label className="compact-toggle"><input type="checkbox" checked={detector.enabled ?? false} onChange={(event) => update("enabled", event.target.checked)} /><span>Detection enabled</span></label>
      <label>OpenVINO model<input value={detector.model_path || detector.model_xml || ""} onChange={(event) => { update("model_path", event.target.value); update("model_xml", ""); }} /><small>Server path to the model. An adjacent model-proc file is discovered automatically.</small></label>
      <label>Device<input value={detector.device || "CPU"} onChange={(event) => update("device", event.target.value)} placeholder="GPU" /></label>
      <label>Labels file<input value={detector.labels_path || ""} onChange={(event) => update("labels_path", event.target.value)} /><small>Optional server path for custom model labels.</small></label>
      <label>Default incident eligibility<select value={String(detector.require_incident_zone ?? true)} onChange={(event) => update("require_incident_zone", event.target.value === "true")}><option value="true">Incident zones only</option><option value="false">Zones and full frame</option></select><small>Cameras can override this. Ignore zones always apply.</small></label>
    </div>
    <section className="sub-panel">
      <h3>High-resolution incident confirmation</h3>
      <label className="check-field"><input type="checkbox" checked={detector.native?.verification_enabled ?? true} onChange={event => update("native.verification_enabled", event.target.checked)} /> Verify objects in main-recording crops before creating incidents</label>
      <p>When disabled, substream detections validate incidents and usable aligned main-recording images are still promoted, without requiring object confirmation. Projected boxes are not marked as main-stream verified.</p>
      <p>When enabled, a clear matching detection confirms an object; three clear misses reject it. Missing or unclear evidence remains unverified and does not send an alert. Verification waits for recorded frames, so alerts arrive later. Live detection and tracking continue while it waits.</p>
    </section>
    <NativeBudgetSettings values={detector.native?.budget || {}} onChange={(key, value) => update(`native.budget.${key}`, value)} />
    <details className="tracking-class-picker">
      <summary>Tracked classes: {trackedClasses === null ? "All model classes" : trackedClasses.length ? trackedClasses.join(", ") : "None"}</summary>
      <div className="form-grid" role="group" aria-label="Tracked classes">
        <label className="compact-toggle"><input type="checkbox" checked={trackedClasses === null} onChange={(event) => update("native.tracking_classes", event.target.checked ? null : [])} /><span>All model classes</span></label>
        {classes.map((label) => <label className="compact-toggle" key={label}><input type="checkbox" aria-label={`Track ${label}`} checked={trackedClasses === null || trackedClasses.includes(label)} onChange={(event) => {
          const current = trackedClasses ?? classes;
          update("native.tracking_classes", event.target.checked ? [...new Set([...current, label])] : current.filter((value) => value !== label));
        }} /><span>{label}</span></label>)}
      </div>
      {!classes.length ? <p>Choose a model or add a class under Per-class incident thresholds to populate this list.</p> : null}
      <small>Only selected classes receive tracking IDs and create native incidents. The model still evaluates all its output classes. Save to restart native tracking with this selection.</small>
    </details>
    {numericFields("detection")}
    <h3>Stationary object policy</h3>
    <label className="compact-toggle"><input type="checkbox" checked={detector.native?.stationary?.enabled ?? true} onChange={(event) => update("native.stationary.enabled", event.target.checked)} /><span>Require movement for selected classes</span></label>
    <label>Classes requiring movement<input value={labelsText} onChange={(event) => { setLabelsText(event.target.value); update("native.stationary.labels", [...new Set(event.target.value.split(",").map((label) => label.trim().toLowerCase()).filter(Boolean))]); }} /><small>Comma-separated model labels. Other classes keep presence alerts, even while standing still.</small></label>
    {numericFields("stationary")}
    <p>Selected classes must show movement before starting an incident. Stationary objects remain tracked as scene context. Reconnecting a stream resets IDs and movement evidence.</p>
    <details><summary>Per-class incident thresholds</summary>
      <p>Leave an override blank to use the global value. Camera zones can also set confidence thresholds.</p>
      <div className="form-grid"><label>Additional model class<input value={extraClass} onChange={(event) => setExtraClass(event.target.value)} /></label><button type="button" disabled={!extraClass.trim()} onClick={() => { setAddedClasses((current) => [...current, extraClass.trim().toLowerCase()]); setExtraClass(""); }}>Add class</button></div>
      {classes.map((label) => <div className="form-grid" key={label}>
        <label>{label} confidence<input aria-label={`${label} confidence`} type="number" min="0.01" max="0.99" step="0.01" placeholder={`Global (${detector.confidence_threshold ?? 0.45})`} value={detector.event_class_confidence_thresholds?.[label] ?? ""} onChange={(event) => override("event_class_confidence_thresholds", label, event.target.value)} /></label>
        <label>{label} confirmation frames<input type="number" min="1" max="5" step="1" placeholder={`Global (${detector.event_confirmation_frames ?? 2})`} value={detector.event_class_confirmation_frames?.[label] ?? ""} onChange={(event) => override("event_class_confirmation_frames", label, event.target.value)} /></label>
      </div>)}
    </details>
    <details><summary>Advanced native pipeline settings</summary>{numericFields("advanced")}<p>Inference requests and execution streams belong to the model shared across cameras. These changes reload native capture when saved.</p></details>
    {error ? <p role="alert" className="error-banner">{error}</p> : null}
  </section>;
}
