import React, { useEffect, useRef, useState } from "react";
import { Activity, Cpu, Gauge, X } from "lucide-react";
import { appUrl, fetch } from "../shared/api.js";
import { modelEvaluationCoverage } from "../modelEvaluation.mjs";

function normalizeDetectorModelPath(value) {
  return String(value || "").replaceAll("\\", "/").replace(/\/+/g, "/");
}

function findDetectorModel(models, activePath) {
  const active = normalizeDetectorModelPath(activePath);
  if (!active) return undefined;
  return (models || []).find((model) => {
    const path = normalizeDetectorModelPath(model.path);
    return path === active || path.endsWith(active) || active.endsWith(path);
  });
}

export function ModelsAndHardwarePanel({ config, updateConfig, detectorModels = [], accelerator = null }) {
  const activeModelPath = config?.detector?.model_path || config?.detector?.model_xml || "";
  const validEvaluationModels = detectorModels.filter((model) => model.valid).sort((left, right) => String(left.path).localeCompare(String(right.path)));
  const defaultBaselinePath = validEvaluationModels.filter((model) => String(model.path) < activeModelPath).at(-1)?.path
    || validEvaluationModels.filter((model) => model.path !== activeModelPath).at(-1)?.path
    || "";
  const [modelEvaluationDraft, setModelEvaluationDraft] = useState({ baseline_path: "", candidate_path: "", sample_count: 200, confidence: 0.25 });
  const [modelEvaluation, setModelEvaluation] = useState({ status: "idle" });
  const [modelEvaluationError, setModelEvaluationError] = useState("");
  const [modelEvaluationPreview, setModelEvaluationPreview] = useState(null);
  const modelEvaluationDialogRef = useRef(null);
  const modelEvaluationTriggerRef = useRef(null);
  const openvinoDevices = accelerator?.openvino_devices || [];
  const hasOpenvinoGpu = openvinoDevices.includes("GPU");
  const detectorBackend = config?.detector?.backend || "openvino";
  const coremlLabel = accelerator?.is_macos
    ? accelerator?.coreml_available ? "Core ML available" : "Core ML not installed"
    : "Core ML is macOS only";
  const gpuLabel = accelerator?.is_apple_silicon
    ? "Mac GPU detected, OpenVINO GPU not available on Apple GPU"
    : accelerator?.has_nvidia
      ? "NVIDIA GPU detected"
      : hasOpenvinoGpu
        ? "OpenVINO GPU device available"
        : "No OpenVINO GPU device reported";
  const ffmpegAcceleration = accelerator?.ffmpeg_hardware_acceleration || {};
  const vaapi = ffmpegAcceleration.vaapi || {};
  const qsv = ffmpegAcceleration.qsv || {};
  const vaapiLabel = vaapi.available
    ? `VAAPI available (${(vaapi.encoders || []).join(", ") || "encoders detected"})`
    : vaapi.listed
      ? "VAAPI listed by FFmpeg but runtime init failed"
      : "VAAPI not available to FFmpeg";
  const qsvLabel = qsv.available
    ? `Intel QSV available (${(qsv.encoders || []).join(", ") || "encoders detected"})`
    : qsv.listed
      ? "Intel QSV listed by FFmpeg but runtime init failed"
      : "Intel QSV not available to FFmpeg";
  const activeModel = findDetectorModel(detectorModels, activeModelPath);
  const evaluationCoverage = modelEvaluationCoverage(modelEvaluation.result);

  function closeModelEvaluationPreview() {
    setModelEvaluationPreview(null);
    window.requestAnimationFrame(() => modelEvaluationTriggerRef.current?.isConnected && modelEvaluationTriggerRef.current.focus());
  }

  useEffect(() => {
    if (!modelEvaluationPreview) return undefined;
    const dialog = modelEvaluationDialogRef.current;
    const focusable = () => [...(dialog?.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])') || [])];
    window.requestAnimationFrame(() => focusable()[0]?.focus());
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeModelEvaluationPreview();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [modelEvaluationPreview]);

  useEffect(() => {
    setModelEvaluationDraft((current) => ({
      ...current,
      candidate_path: current.candidate_path || activeModelPath,
      baseline_path: current.baseline_path || defaultBaselinePath,
    }));
  }, [activeModelPath, defaultBaselinePath]);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const load = async () => {
      try {
        const response = await fetch("/api/detector/model-evaluation", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        if (cancelled) return;
        setModelEvaluation(payload);
        if (["queued", "running", "cancelling"].includes(payload.status)) timer = window.setTimeout(load, 1_000);
      } catch {
        // Optional configuration telemetry.
      }
    };
    void load();
    return () => { cancelled = true; if (timer) window.clearTimeout(timer); };
  }, [modelEvaluation.status]);

  async function startModelEvaluation() {
    setModelEvaluationError("");
    try {
      const response = await fetch("/api/detector/model-evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelEvaluationDraft),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to start model evaluation.");
      setModelEvaluation(payload);
    } catch (error) {
      setModelEvaluationError(error.message || "Unable to start model evaluation.");
    }
  }

  async function cancelModelEvaluation() {
    const response = await fetch("/api/detector/model-evaluation", { method: "DELETE" });
    const payload = await response.json().catch(() => ({}));
    if (response.ok) setModelEvaluation(payload);
    else setModelEvaluationError(payload.detail || "Unable to cancel model evaluation.");
  }

  return (
    <section className="telemetry-section models-hardware-panel">
      <div className="telemetry-section-head"><div><h3 className="section-heading-with-icon"><span className="section-heading-icon"><Cpu size={16} /></span>Models &amp; Hardware</h3><p>Compare detector models and inspect accelerator readiness.</p></div></div>
      <section className="detection-settings-card wide-card model-evaluation-card">
        <header className="detection-settings-card-head">
          <div className="detection-settings-card-icon"><Gauge size={18} /></div>
          <div><h3>Model Evaluation</h3><p>Compare two models on the same recent clean incident images without changing production detection.</p></div>
        </header>
        <div className="detection-field-grid">
          <label>Baseline model<select value={modelEvaluationDraft.baseline_path} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, baseline_path: event.target.value }))}>
            <option value="">Select model</option>
            {detectorModels.filter((model) => model.valid).map((model) => <option key={model.path} value={model.path}>{String(model.path).split("/").slice(-2, -1)[0] || model.name}</option>)}
          </select></label>
          <label>Candidate model<select value={modelEvaluationDraft.candidate_path} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, candidate_path: event.target.value }))}>
            <option value="">Select model</option>
            {detectorModels.filter((model) => model.valid).map((model) => <option key={model.path} value={model.path}>{String(model.path).split("/").slice(-2, -1)[0] || model.name}{model.path === activeModelPath ? " (active)" : ""}</option>)}
          </select></label>
          <label>Recent images<input type="number" min="10" max="500" step="10" value={modelEvaluationDraft.sample_count} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, sample_count: Number(event.target.value) }))} /><small>Round-robin sampled across cameras; 200 is a useful first pass.</small></label>
          <label>Candidate threshold<input type="number" min="0.01" max="0.99" step="0.01" value={modelEvaluationDraft.confidence} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, confidence: Number(event.target.value) }))} /><small>Use the same low evidence threshold for both models.</small></label>
        </div>
        <div className="model-evaluation-actions">
          <span className="admin-action-kind">Background task · does not change production</span>
          <button type="button" className="primary" onClick={() => void startModelEvaluation()} disabled={!modelEvaluationDraft.baseline_path || !modelEvaluationDraft.candidate_path || ["queued", "running", "cancelling"].includes(modelEvaluation.status)}><Activity size={15} />Run comparison</button>
          {["queued", "running", "cancelling"].includes(modelEvaluation.status) ? <button type="button" onClick={() => void cancelModelEvaluation()} disabled={modelEvaluation.status === "cancelling"}><X size={15} />Cancel</button> : null}
          <span className={`model-evaluation-state ${modelEvaluation.status}`}>{String(modelEvaluation.status || "idle").replaceAll("_", " ")}{modelEvaluation.progress?.total ? ` · ${modelEvaluation.progress.completed}/${modelEvaluation.progress.total}` : ""}</span>
        </div>
        {modelEvaluationError || modelEvaluation.error ? <div className="error-banner">{modelEvaluationError || modelEvaluation.error}</div> : null}
        {modelEvaluation.result ? <div className="model-evaluation-results">
          <div className="model-evaluation-summary">
            <span><strong>{evaluationCoverage.compared} / {evaluationCoverage.total}</strong> images compared</span>
            <span><strong>{modelEvaluation.result.camera_count}</strong> cameras</span>
            <span><strong>{modelEvaluation.result.disagreement_frames}</strong> disagreements</span>
            <span><strong>{modelEvaluation.result.candidate.average_ms} ms</strong> candidate average</span>
          </div>
          {!evaluationCoverage.complete ? <div className="probe-result bad">
            <strong>{evaluationCoverage.compared ? "Partial comparison" : "No valid comparison"}</strong>
            <span>{evaluationCoverage.failed} images could not be compared. Baseline errors: {evaluationCoverage.baselineErrors}; candidate errors: {evaluationCoverage.candidateErrors}.</span>
            {evaluationCoverage.compared ? <span>Agreement and recall include only the {evaluationCoverage.compared} images successfully evaluated by both models.</span> : null}
          </div> : null}
          {modelEvaluation.result.disagreements?.length ? <div className="model-evaluation-disagreements">
            {modelEvaluation.result.disagreements.map((item) => <article key={`${item.source_kind}-${item.source_id}`}>
              <button type="button" className="model-evaluation-image-button" onClick={(event) => { modelEvaluationTriggerRef.current = event.currentTarget; setModelEvaluationPreview(item); }} aria-label={`Enlarge ${item.camera_id} comparison image`}><img src={appUrl(item.image_url)} alt="" loading="lazy" /></button>
              <span><strong>{item.camera_id}</strong><small>{item.source_kind === "motion_audit" ? "Motion audit negative" : "Incident"} · {item.created_at}</small></span>
            </article>)}
          </div> : evaluationCoverage.complete ? <div className="probe-result ok"><strong>No label disagreements</strong><span>Both models returned the same label sets on this corpus.</span></div> : evaluationCoverage.compared > 0 ? <div className="probe-result"><strong>No label disagreements among successful comparisons</strong><span>Failed images were excluded.</span></div> : null}
        </div> : null}
        {modelEvaluationPreview ? <div className="model-evaluation-preview" role="presentation">
          <button type="button" className="live-overlay-backdrop" onClick={closeModelEvaluationPreview} aria-label="Close comparison image" />
          <section ref={modelEvaluationDialogRef} role="dialog" aria-modal="true" aria-labelledby="model-evaluation-preview-title">
            <header><div><strong id="model-evaluation-preview-title">{modelEvaluationPreview.camera_id}</strong><small>{modelEvaluationPreview.created_at}</small></div><button type="button" className="icon-only" onClick={closeModelEvaluationPreview} aria-label="Close comparison image"><X size={19} /></button></header>
            <img src={appUrl(modelEvaluationPreview.image_url)} alt={`${modelEvaluationPreview.camera_id} model comparison source`} />
          </section>
        </div> : null}
      </section>
      <details className="detection-settings-card detection-feature-card diagnostics-card wide-card" open>
        <summary><span className="detection-settings-card-icon"><Cpu size={18} /></span><span><strong>Model &amp; accelerator diagnostics</strong><small>Loaded model metadata and available processing hardware.</small></span></summary>
        <div className="detection-feature-body diagnostics-grid">
          {activeModel ? (
            <div className={`probe-result ${activeModel.valid ? "ok" : "bad"}`}>
              <strong>{activeModel.valid ? "OpenVINO IR ready" : "OpenVINO IR incomplete"}</strong>
              <span>XML: {activeModel.path}</span>
              <span>Weights: {activeModel.bin_present ? activeModel.bin_path : "matching .bin file not found"}</span>
              <span>Input: {activeModel.input_shape.join(" x ") || "unknown"}</span>
              <span>Output: {activeModel.output_shapes.map((shape) => shape.join(" x ")).join(", ") || "unknown"}</span>
              <span>Task: {activeModel.task || "detect"}</span>
              <span>Classes: {activeModel.classes.join(", ") || "none found"}</span>
              {activeModel.error ? <span>{activeModel.error}</span> : null}
            </div>
          ) : null}
          {detectorBackend === "coreml" ? (
            <div className="detection-field-grid">
              <label>Core ML Model Path<input value={config.detector?.coreml_model_path || ""} onChange={(event) => updateConfig(["detector", "coreml_model_path"], event.target.value)} placeholder="model.mlpackage or model.mlmodel" /></label>
            </div>
          ) : null}
          <div className="probe-result">
            <strong>Accelerator</strong>
            <span>System: {accelerator ? `${accelerator.system} ${accelerator.machine}` : "checking..."}</span>
            <span>Detector recommendation: {accelerator?.recommended_detector_backend === "coreml" ? "Core ML" : "OpenVINO / ONNX"}</span>
            <span>{coremlLabel}</span>
            <span>OpenVINO devices: {openvinoDevices.length ? openvinoDevices.join(", ") : "none reported"}</span>
            <span>{gpuLabel}</span>
            <span>FFmpeg acceleration: {ffmpegAcceleration.configured || config.hardware_acceleration || "auto"}</span>
            <span>FFmpeg: {ffmpegAcceleration.ffmpeg_path || accelerator?.ffmpeg_path || config.ffmpeg_path || "ffmpeg"}</span>
            <span>FFprobe: {ffmpegAcceleration.ffprobe_path || accelerator?.ffprobe_path || "ffprobe"}</span>
            <span>FFplay: {ffmpegAcceleration.ffplay_path || accelerator?.ffplay_path || "ffplay"}</span>
            <span>{vaapiLabel}</span>
            {vaapi.render_devices?.length ? <span>VAAPI render devices: {vaapi.render_devices.join(", ")}</span> : null}
            {vaapi.filters?.length ? <span>VAAPI filters: {vaapi.filters.join(", ")}</span> : null}
            {vaapi.runtime_error ? <span>VAAPI runtime: {vaapi.runtime_error}</span> : null}
            <span>{qsvLabel}</span>
            {qsv.render_devices?.length ? <span>QSV render devices: {qsv.render_devices.join(", ")}</span> : null}
            {qsv.decoders?.length ? <span>QSV decoders: {qsv.decoders.join(", ")}</span> : null}
            {qsv.runtime_error ? <span>QSV runtime: {qsv.runtime_error}</span> : null}
            {accelerator?.recommended_openvino_device ? <span>Recommended OpenVINO device: {accelerator.recommended_openvino_device}</span> : null}
            {accelerator?.coreml_error ? <span>{accelerator.coreml_error}</span> : null}
            {accelerator?.openvino_error ? <span>{accelerator.openvino_error}</span> : null}
          </div>
        </div>
      </details>
    </section>
  );
}
