import NativeBudgetSettings from "./NativeBudgetSettings.jsx";
import "./nativeDetection.css";
import { Cpu, ScanLine, Route, Gauge, SlidersHorizontal } from "lucide-react";
import { nextTabId } from "../adminWorkspace.mjs";
import React, { useEffect, useState } from "react";
import { DEFAULT_STATIONARY_LABELS, NATIVE_DETECTION_FIELDS, detectionFieldValue, nativeDetectionError } from "../nativeDetectionSettings.mjs";

const SECTIONS = [
  { id: "model", label: "Model", icon: Cpu, help: "Choose the detection model and the hardware that runs it." },
  { id: "incidents", label: "Incidents", icon: ScanLine, help: "Decide which detections become incidents and how they are confirmed." },
  { id: "tracking", label: "Tracking", icon: Route, help: "Choose objects to follow and when movement is required for an incident." },
  { id: "performance", label: "Performance", icon: Gauge, help: "Balance detection frequency, responsiveness, and resource use." },
  { id: "advanced", label: "Advanced", icon: SlidersHorizontal, help: "Tune shared inference resources and recovery limits. Change these only when needed." },
];

function SettingsCard({ title, description, children }) {
  return <section className="detection-settings-card native-settings-card">
    <header><h3>{title}</h3>{description ? <p>{description}</p> : null}</header>
    {children}
  </section>;
}

export function NativeDetectionSettings({ detector, updateConfig, modelClasses = [], section: selectedSection, onSectionChange }) {
  const [localSection, setLocalSection] = useState("model");
  const section = SECTIONS.some(item => item.id === (selectedSection ?? localSection)) ? (selectedSection ?? localSection) : "model";
  const selectSection = onSectionChange || setLocalSection;
  const active = SECTIONS.find(item => item.id === section);
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
  const [classSearch, setClassSearch] = useState("");
  const visibleClasses = classes.filter(label => label.toLowerCase().includes(classSearch.toLowerCase().trim()));
  const trackedClasses = detector.native?.tracking_classes ?? null;
  const update = (path, value) => updateConfig(["detector", ...path.split(".")], value);
  const numericFields = (...paths) => <div className="form-grid">{NATIVE_DETECTION_FIELDS.filter((field) => paths.includes(field.path)).map((field) => <label key={field.path}>{field.label}<input type="number" aria-label={field.label} min={field.min} max={field.max} step={field.step} value={detectionFieldValue(detector, field)} onChange={(event) => update(field.path, event.target.value === "" ? "" : Number(event.target.value))} />{field.help ? <small>{field.path === "live_sample_fps" ? "Frames sampled from each camera. Adaptive mode uses its idle and active rates; fixed mode divides this rate by the inference interval." : field.help}</small> : null}</label>)}</div>;
  function override(key, label, value) {
    const next = { ...detector[key] };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    update(key, next);
  }
  const error = nativeDetectionError(detector);
  return <section className="native-detection-workspace subsection-workspace">
    <div className="admin-section-tabs camera-section-tabs detection-subsection-tabs" role="tablist" aria-label="Detection settings sections" onKeyDown={(event) => {
      const next = nextTabId(SECTIONS.map(item => item.id), section, event.key);
      if (!next) return;
      event.preventDefault();
      selectSection(next);
      event.currentTarget.querySelector(`#native-detection-tab-${next}`)?.focus();
    }}>
      {SECTIONS.map(({ id, label, icon: Icon }) => <button key={id} id={`native-detection-tab-${id}`} type="button" role="tab" aria-selected={section === id} aria-controls={`native-detection-panel-${id}`} tabIndex={section === id ? 0 : -1} className={section === id ? "active" : ""} onClick={() => selectSection(id)}><Icon size={15} />{label}</button>)}
    </div>
    <div className="subsection-workspace-content">
      <header className="native-settings-intro"><div><h3>{active.label}</h3><p>{active.help}</p></div><span>Global defaults · Save to apply</span></header>
      {error ? <p role="alert" className="error-banner">{error} Check the relevant tab before saving.</p> : null}
      {SECTIONS.map(({ id }) => <div key={id} id={`native-detection-panel-${id}`} role="tabpanel" aria-labelledby={`native-detection-tab-${id}`} hidden={section !== id} className="native-settings-panes">
        {id === "model" ? <>
          <SettingsCard title="Detection engine" description="Enable object detection for your cameras. Camera settings control individual streams and zones.">
            <label className="compact-toggle"><input type="checkbox" checked={detector.enabled ?? false} onChange={(event) => update("enabled", event.target.checked)} /><span>Detection enabled</span></label>
            {!detector.enabled ? <p className="native-settings-note">Detection is off. You can prepare settings here before enabling it.</p> : null}
          </SettingsCard>
          <SettingsCard title="Model and device" description="Use an OpenVINO model installed on this server.">
            <div className="form-grid">
              <label>OpenVINO model<input value={detector.model_path || detector.model_xml || ""} onChange={(event) => { update("model_path", event.target.value); update("model_xml", ""); }} /><small>Server path to the model. An adjacent model-proc file is discovered automatically.</small></label>
              <label>Device<input value={detector.device || "CPU"} onChange={(event) => update("device", event.target.value)} placeholder="GPU" /></label>
              <label>Labels file<input value={detector.labels_path || ""} onChange={(event) => update("labels_path", event.target.value)} /><small>Optional server path for custom model labels.</small></label>
            </div>
          </SettingsCard>
        </> : null}
        {id === "incidents" ? <>
          <SettingsCard title="Incident rules" description="Set the minimum evidence needed to start an incident. Higher confidence and more confirmation frames reduce weak detections, but may miss brief appearances.">
              <label>Default incident eligibility<select value={String(detector.require_incident_zone ?? true)} onChange={(event) => update("require_incident_zone", event.target.value === "true")}><option value="true">Incident zones only</option><option value="false">Zones and full frame</option></select><small>Cameras can override this. Ignore zones always apply.</small></label>

            {numericFields("confidence_threshold", "event_confirmation_frames", "native.activity_timeout_seconds")}
          </SettingsCard>
          <SettingsCard title="High-resolution incident confirmation" description="Check recording detail before alerting.">
              <label className="check-field"><input type="checkbox" checked={detector.native?.verification_enabled ?? true} onChange={event => update("native.verification_enabled", event.target.checked)} /> Verify objects in main-recording crops before creating incidents</label>
              {!(detector.native?.verification_enabled ?? true) ? <p>When disabled, substream detections validate incidents and usable aligned main-recording images are still promoted, without requiring object confirmation. Projected boxes are not marked as main-stream verified.</p> : <p>When enabled, a clear matching detection confirms an object; three clear misses reject it. Missing or unclear evidence remains unverified and does not send an alert. Verification waits for recorded frames, so alerts arrive later. Live detection and tracking continue while it waits.</p>}
          </SettingsCard>
          <SettingsCard title="Per-class incident thresholds" description="Optional exceptions for individual object classes.">
            <details><summary>Customize class thresholds</summary>
              <p>Leave an override blank to use the global value. Camera zones can also set confidence thresholds.</p>
              <div className="form-grid"><label>Additional model class<input value={extraClass} onChange={(event) => setExtraClass(event.target.value)} /></label><button type="button" disabled={!extraClass.trim()} onClick={() => { setAddedClasses((current) => [...current, extraClass.trim().toLowerCase()]); setExtraClass(""); }}>Add class</button></div>
              <label>Find class thresholds<input type="search" value={classSearch} onChange={event => setClassSearch(event.target.value)} placeholder="Filter classes…" /></label>
              {!visibleClasses.length ? <p>No matching classes. Add a model class above.</p> : null}
              <div className="native-class-list">
              {visibleClasses.map((label) => <div className="form-grid" key={label}>
                <label>{label} confidence<input aria-label={`${label} confidence`} type="number" min="0.01" max="0.99" step="0.01" placeholder={`Global (${detector.confidence_threshold ?? 0.45})`} value={detector.event_class_confidence_thresholds?.[label] ?? ""} onChange={(event) => override("event_class_confidence_thresholds", label, event.target.value)} /></label>
                <label>{label} confirmation frames<input type="number" min="1" max="5" step="1" placeholder={`Global (${detector.event_confirmation_frames ?? 2})`} value={detector.event_class_confirmation_frames?.[label] ?? ""} onChange={(event) => override("event_class_confirmation_frames", label, event.target.value)} /></label>
              </div>)}
              </div>
            </details>
          </SettingsCard>
        </> : null}
        {id === "tracking" ? <>
          <SettingsCard title="Objects to track" description="Choose which model classes can receive tracking IDs and create incidents.">
              <div className="tracking-class-picker">
                <p>Tracked classes: {trackedClasses === null ? "All model classes" : trackedClasses.length ? trackedClasses.join(", ") : "None"}</p>
                <label>Find tracked classes<input type="search" value={classSearch} onChange={event => setClassSearch(event.target.value)} placeholder="Filter classes…" /></label>
                <div className="form-grid native-class-list" role="group" aria-label="Tracked classes">
                  <label className="compact-toggle"><input type="checkbox" checked={trackedClasses === null} onChange={(event) => update("native.tracking_classes", event.target.checked ? null : [])} /><span>All model classes</span></label>
                  {visibleClasses.map((label) => <label className="compact-toggle" key={label}><input type="checkbox" aria-label={`Track ${label}`} checked={trackedClasses === null || trackedClasses.includes(label)} onChange={(event) => {
                    const current = trackedClasses ?? classes;
                    update("native.tracking_classes", event.target.checked ? [...new Set([...current, label])] : current.filter((value) => value !== label));
                  }} /><span>{label}</span></label>)}
                </div>
                {classes.length > 0 && !visibleClasses.length ? <p>No matching classes.</p> : null}
                {!classes.length ? <p>Choose a model or add a class in Incidents → Per-class incident thresholds to populate this list.</p> : null}
                <small>Only selected classes receive tracking IDs and create native incidents. The model still evaluates all its output classes. Save to restart native tracking with this selection.</small>
              </div>
          </SettingsCard>
          <SettingsCard title="Stationary object policy" description="Avoid repeated incidents from parked vehicles while keeping presence alerts for other objects.">
              <label className="compact-toggle"><input type="checkbox" checked={detector.native?.stationary?.enabled ?? true} onChange={(event) => update("native.stationary.enabled", event.target.checked)} /><span>Require movement for selected classes</span></label>
              <label>Classes requiring movement<input value={labelsText} onChange={(event) => { setLabelsText(event.target.value); update("native.stationary.labels", [...new Set(event.target.value.split(",").map((label) => label.trim().toLowerCase()).filter(Boolean))]); }} /><small>Comma-separated model labels. Other classes keep presence alerts, even while standing still.</small></label>
              {numericFields("native.stationary.stationary_seconds")}
              <details><summary>Movement sensitivity</summary>
                <p>Adjust these together if stationary objects are mistaken for moving objects.</p>
                {numericFields("native.stationary.window_seconds", "native.stationary.moving_threshold", "native.stationary.stationary_threshold")}
              </details>
              <p>Selected classes must show movement before starting an incident. Stationary objects remain tracked as scene context. Reconnecting a stream resets IDs and movement evidence.</p>
          </SettingsCard>
        </> : null}
        {id === "performance" ? <>
          <SettingsCard title="Frame sampling" description="Set how frequently camera frames enter detection. More frames use more resources.">
            {numericFields("live_sample_fps")}
            <details><summary>Fixed-rate inference</summary><p>The inference interval applies when adaptive inference is disabled.</p>{numericFields("native.inference_interval")}</details>
          </SettingsCard>
          <NativeBudgetSettings grouped values={detector.native?.budget || {}} onChange={(key, value) => update(`native.budget.${key}`, value)} />
        </> : null}
        {id === "advanced" ? <>
          <SettingsCard title="Shared model resources" description="These resources are shared across cameras. Saving changes reloads native capture.">
            {numericFields("native.batch_size", "native.inference_requests", "native.inference_streams")}
          </SettingsCard>
          <SettingsCard title="Result quality and recovery" description="Limit stale results and recover when detection metadata stops arriving.">
            {numericFields("native.maximum_observation_age_seconds", "native.metadata_restart_seconds")}
          </SettingsCard>
          <SettingsCard title="Tracking capacity and overlap" description="Adjust object capacity and model overlap suppression.">
            {numericFields("native.maximum_tracks", "nms_threshold")}
          </SettingsCard>
        </> : null}
      </div>)}
    </div>
  </section>;
}
