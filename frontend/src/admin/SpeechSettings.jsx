import React, { useEffect, useState } from "react";
import { AudioLines, CircleAlert, Mic, RefreshCcw } from "lucide-react";
import { fetch } from "../shared/api.js";
import {
  WHISPER_MODELS,
  cameraSpeechSettings,
  resolvedWhisperModel,
  speechFailureLabel,
  speechListenerLabel,
  speechSettings,
  withSetting,
} from "../speechSettings.mjs";

const STATUS_REFRESH_MS = 10000;

function useSpeechStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  async function load() {
    try {
      const response = await fetch("/api/speech/status");
      if (!response.ok) throw new Error("Speech status is unavailable.");
      setStatus(await response.json());
      setError("");
    } catch (reason) {
      setError(reason.message || "Speech status is unavailable.");
    }
  }
  useEffect(() => {
    void load();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, STATUS_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, []);
  return { status, error, reload: load };
}

function ModelSummary({ speech, models }) {
  if (!models) return null;
  const selected = resolvedWhisperModel(speech);
  const installed = models.whisper_installed || [];
  const vadText = models.vad_installed ? "Voice model installed" : "Voice model missing";
  const whisperText = installed.length ? `Whisper installed: ${installed.join(", ")}` : "No Whisper models installed";
  const selectedMissing = speech.whisper_models_dir && !installed.includes(selected);
  return <div className={`speech-model-summary${models.vad_installed ? "" : " missing"}`}>
    <span className={models.vad_installed ? "ok" : "missing"}>{vadText}</span>
    <span className={installed.length ? "ok" : "missing"}>{whisperText}</span>
    {selectedMissing ? <span className="missing">Selected model {selected} is not installed</span> : null}
    {!models.vad_installed || !installed.length ? <small>Install with <code>install-docker-models.sh --speech --device {speech.transcription_device}</code> (add <code>--native</code> for non-Docker installs).</small> : null}
  </div>;
}

function ServiceSummary({ status, error, reload }) {
  if (error) return <p className="settings-help" role="alert"><CircleAlert size={14} /> {error} <button type="button" className="link-button" onClick={() => void reload()}><RefreshCcw size={13} />Retry</button></p>;
  if (!status) return <p className="settings-help">Loading speech status…</p>;
  const transcriber = status.transcriber;
  const parts = [`Service: ${status.state || "unknown"}`];
  if (status.reason) parts.push(status.reason);
  if (transcriber) {
    parts.push(`Transcriber: ${transcriber.state}${transcriber.device ? ` on ${transcriber.device}` : ""}${transcriber.model ? ` (${transcriber.model})` : ""}`);
    if (transcriber.reason) parts.push(transcriber.reason);
    if (transcriber.queue_depth) parts.push(`${transcriber.queue_depth} waiting`);
    if (transcriber.dropped) parts.push(`${transcriber.dropped} dropped`);
  }
  return <p className="settings-help speech-service-summary">{parts.join(" · ")}</p>;
}

export function SpeechSettings({ config, updateConfig }) {
  const { status, error, reload } = useSpeechStatus();
  const speech = speechSettings(config);
  const cameras = config?.cameras || [];
  const setSpeech = (key, value) => updateConfig(["speech"], withSetting(speech, key, value));
  const setCameraSpeech = (index, key, value) => {
    const current = cameraSpeechSettings(cameras[index]);
    const next = withSetting(current, key, value);
    if (key === "enabled" && !value) next.transcribe = false;
    updateConfig(["cameras", index, "speech"], next);
  };
  const listeningCount = cameras.filter((camera) => camera?.speech?.enabled).length;
  const autoLabel = `Automatic (${resolvedWhisperModel({ ...speech, transcription_model: "auto" })})`;

  return <>
    <section className="detection-settings-card primary">
      <header className="detection-settings-card-head">
        <div className="detection-settings-card-icon"><Mic size={18} /></div>
        <div><h3>Speech detection</h3><p>Listen for people talking and optionally transcribe English speech on this server. Audio is never stored separately; playback uses the recording.</p></div>
        <label className="compact-toggle"><input type="checkbox" checked={speech.enabled} onChange={(event) => setSpeech("enabled", event.target.checked)} /><span>Speech enabled</span></label>
      </header>
      <div className="speech-consent-notice" role="note"><CircleAlert size={16} /><span>Recording or transcribing conversations is regulated in many places (all-party consent, GDPR, workplace rules) and is often treated more strictly than video. Make sure it is lawful where your cameras are.</span></div>
      <ModelSummary speech={speech} models={status?.models} />
      <div className="detection-field-grid">
        <label>Transcription device<select value={speech.transcription_device} onChange={(event) => setSpeech("transcription_device", event.target.value)}>
          <option value="CPU">CPU</option>
          <option value="GPU">GPU (falls back to CPU)</option>
        </select><small>Voice detection always runs on the CPU; only Whisper uses this device.</small></label>
        <label>Whisper model<select value={speech.transcription_model} onChange={(event) => setSpeech("transcription_model", event.target.value)}>
          <option value="auto">{autoLabel}</option>
          {WHISPER_MODELS.map((model) => <option key={model} value={model}>{model}{status?.models && !(status.models.whisper_installed || []).includes(model) ? " (not installed)" : ""}</option>)}
        </select><small>Automatic uses base.en on CPU and small.en on GPU. Larger models are more accurate and slower.</small></label>
        <label>Whisper CPU threads<input type="number" min="0" max="64" step="1" value={speech.transcription_cpu_threads} onChange={(event) => setSpeech("transcription_cpu_threads", Math.max(0, Number(event.target.value) || 0))} /><small>0 = automatic (half the cores, up to 4). Lower leaves more CPU for cameras; transcription also runs at reduced priority.{status?.transcriber?.cpu_threads ? ` Currently ${status.transcriber.cpu_threads}.` : ""}</small></label>
        <label>Keep transcripts<input type="number" min="1" max="365" step="1" value={speech.retention_days} onChange={(event) => setSpeech("retention_days", Number(event.target.value))} /><small>Days. Independent of video retention.</small></label>
        <label>Queue per camera<input type="number" min="1" max="64" step="1" value={speech.max_queue_per_camera} onChange={(event) => setSpeech("max_queue_per_camera", Number(event.target.value))} /><small>Segments waiting for transcription; the oldest is skipped when full.</small></label>
        <label className="compact-toggle wide-field"><input type="checkbox" checked={speech.mqtt_publish_transcripts} onChange={(event) => setSpeech("mqtt_publish_transcripts", event.target.checked)} /><span>Publish transcript text to MQTT</span><small>The Speech on/off sensor is always published for listening cameras. Transcript text leaves SurvNG only when this is on.</small></label>
      </div>
      <details className="detection-compact-details">
        <summary>Model paths</summary>
        <div className="detection-field-grid">
          <label className="wide-field">Voice activity model<input value={speech.vad_model_path} onChange={(event) => setSpeech("vad_model_path", event.target.value)} placeholder="/models/speech/silero_vad_openvino_16k.onnx" /></label>
          <label className="wide-field">Whisper models folder<input value={speech.whisper_models_dir} onChange={(event) => setSpeech("whisper_models_dir", event.target.value)} placeholder="/models/speech/whisper" /><small>Contains whisper-base.en, whisper-small.en, …</small></label>
        </div>
      </details>
      <ServiceSummary status={status} error={error} reload={reload} />
      <p className="settings-help speech-service-summary">Saving speech changes restarts camera workers.</p>
    </section>

    <section className="detection-settings-card wide-card">
      <header className="detection-settings-card-head">
        <div className="detection-settings-card-icon"><AudioLines size={18} /></div>
        <div><h3>Cameras</h3><p>{listeningCount ? `${listeningCount} of ${cameras.length} cameras listening.` : "Choose which cameras listen for speech."}{speech.enabled ? "" : " Speech is turned off above, so no camera listens yet."} A camera listens only while it is on and its Detection switch is on.</p></div>
      </header>
      {cameras.length ? <div className="speech-camera-list">
        {cameras.map((camera, index) => {
          const settings = cameraSpeechSettings(camera);
          const live = status?.cameras?.[camera.id];
          const failure = speechFailureLabel(live?.failure);
          const stateLabel = settings.enabled ? speechListenerLabel(live?.state) : "Not listening";
          return <div className="speech-camera-row" key={camera.id || index}>
            <div className="speech-camera-name"><strong>{camera.name || camera.id}</strong><small className={`speech-camera-state ${live?.state || ""}`}>{stateLabel}{failure && live?.state === "restarting" ? ` · ${failure}` : ""}</small></div>
            <label className="compact-toggle"><input type="checkbox" checked={settings.enabled} onChange={(event) => setCameraSpeech(index, "enabled", event.target.checked)} /><span>Listen</span></label>
            <label className="compact-toggle"><input type="checkbox" checked={settings.transcribe} disabled={!settings.enabled} onChange={(event) => setCameraSpeech(index, "transcribe", event.target.checked)} /><span>Transcribe</span></label>
            <label>Audio from<select value={settings.source} disabled={!settings.enabled} onChange={(event) => setCameraSpeech(index, "source", event.target.value)}>
              <option value="main">Main stream</option>
              <option value="live">Sub stream</option>
            </select></label>
            <label className="speech-camera-incidents">Incidents<select value={settings.incidents} disabled={!settings.enabled} onChange={(event) => setCameraSpeech(index, "incidents", event.target.value)}>
              <option value="attach">Show speech on incidents</option>
              <option value="trigger">Speech also creates incidents</option>
            </select>{settings.enabled && settings.incidents === "trigger" ? <select aria-label={`Notifications for speech incidents on ${camera.name || camera.id}`} value={settings.notify} onChange={(event) => setCameraSpeech(index, "notify", event.target.value)}>
              <option value="motion">Notify like motion-only incidents</option>
              <option value="always">Always notify for speech incidents</option>
              <option value="never">Never notify for speech-only incidents</option>
            </select> : null}{settings.enabled && settings.incidents === "trigger" ? <small>{live?.incident_checks != null ? `${live.incident_checks} checks requested · ${live.incident_checks_suppressed || 0} held back by limits` : "Confirmed speech asks for an incident check."}</small> : null}</label>
            {settings.enabled ? <details className="detection-compact-details speech-camera-tuning">
              <summary>Tuning</summary>
              <div className="detection-field-grid">
                <label>Speech sensitivity<input type="number" min="0.1" max="0.95" step="0.05" value={settings.vad_threshold} onChange={(event) => setCameraSpeech(index, "vad_threshold", Number(event.target.value))} /><small>Lower hears quieter speech but reacts to more noise.</small></label>
                <label>Minimum speech (ms)<input type="number" min="100" max="5000" step="50" value={settings.min_speech_ms} onChange={(event) => setCameraSpeech(index, "min_speech_ms", Number(event.target.value))} /></label>
                <label>End after silence (ms)<input type="number" min="100" max="5000" step="50" value={settings.hangover_ms} onChange={(event) => setCameraSpeech(index, "hangover_ms", Number(event.target.value))} /></label>
                <label>Include before speech (ms)<input type="number" min="0" max="3000" step="50" value={settings.pre_roll_ms} onChange={(event) => setCameraSpeech(index, "pre_roll_ms", Number(event.target.value))} /></label>
                <label>Longest segment (s)<input type="number" min="5" max="120" step="1" value={settings.max_segment_seconds} onChange={(event) => setCameraSpeech(index, "max_segment_seconds", Number(event.target.value))} /></label>
                {settings.incidents === "trigger" ? <>
                  <label>Speech before an incident check (ms)<input type="number" min="300" max="10000" step="100" value={settings.trigger_min_speech_ms} onChange={(event) => setCameraSpeech(index, "trigger_min_speech_ms", Number(event.target.value))} /><small>Voiced time in one utterance before speech asks for a check.</small></label>
                  <label>Time between checks (s)<input type="number" min="0" max="3600" step="5" value={settings.trigger_cooldown_seconds} onChange={(event) => setCameraSpeech(index, "trigger_cooldown_seconds", Number(event.target.value))} /></label>
                  <label>Most checks per 5 minutes<input type="number" min="1" max="60" step="1" value={settings.trigger_max_per_5m} onChange={(event) => setCameraSpeech(index, "trigger_max_per_5m", Number(event.target.value))} /><small>Limits detection work from a TV, radio or long conversation.</small></label>
                </> : null}
              </div>
            </details> : null}
          </div>;
        })}
      </div> : <p className="settings-help">Add a camera first.</p>}
    </section>
  </>;
}
