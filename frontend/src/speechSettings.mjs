// Pure helpers for speech settings and transcript search. Defaults mirror
// SpeechConfig / CameraSpeechConfig in survng/app/config.py.

export const SPEECH_DEFAULTS = Object.freeze({
  enabled: false,
  vad_model_path: "",
  whisper_models_dir: "",
  transcription_device: "CPU",
  transcription_model: "auto",
  transcription_cpu_threads: 0,
  max_queue_per_camera: 8,
  retention_days: 7,
  mqtt_publish_transcripts: false,
});

export const CAMERA_SPEECH_DEFAULTS = Object.freeze({
  enabled: false,
  transcribe: false,
  source: "main",
  vad_threshold: 0.5,
  min_speech_ms: 300,
  hangover_ms: 800,
  pre_roll_ms: 500,
  max_segment_seconds: 30,
});

export const WHISPER_MODELS = Object.freeze(["tiny.en", "base.en", "small.en", "medium.en"]);
const AUTO_MODEL_BY_DEVICE = Object.freeze({ CPU: "base.en", GPU: "small.en" });

export function speechSettings(config) {
  return { ...SPEECH_DEFAULTS, ...(config?.speech || {}) };
}

export function cameraSpeechSettings(camera) {
  return { ...CAMERA_SPEECH_DEFAULTS, ...(camera?.speech || {}) };
}

export function withSetting(settings, key, value) {
  return { ...settings, [key]: value };
}

export function resolvedWhisperModel(settings) {
  const speech = { ...SPEECH_DEFAULTS, ...(settings || {}) };
  if (speech.transcription_model && speech.transcription_model !== "auto") return speech.transcription_model;
  return AUTO_MODEL_BY_DEVICE[String(speech.transcription_device || "CPU").toUpperCase()] || "base.en";
}

const LISTENER_LABELS = Object.freeze({
  starting: "Starting",
  listening: "Listening",
  speaking: "Speech now",
  restarting: "Reconnecting",
  no_audio_track: "No audio track",
  paused: "Paused: camera or detection off",
  stopped: "Stopped",
  disabled: "Off",
  idle: "Off",
  unavailable: "Unavailable",
});

export function speechListenerLabel(state) {
  return LISTENER_LABELS[String(state || "")] || "Not running";
}

const FAILURE_LABELS = Object.freeze({
  stream_unavailable: "Stream unavailable",
  stream_unauthorized: "Stream rejected credentials",
  audio_decode_error: "Audio could not be decoded",
  ffmpeg_error: "FFmpeg error",
  ffmpeg_start_failed: "FFmpeg could not start",
  stream_ended: "Stream ended",
});

export function speechFailureLabel(failure) {
  return FAILURE_LABELS[String(failure || "")] || "";
}

const SEGMENT_STATUS_LABELS = Object.freeze({
  detected: "Speech detected",
  queued: "Waiting to transcribe",
  transcribing: "Transcribing",
  done: "Transcribed",
  no_text: "No words recognized",
  low_confidence: "Low confidence",
  failed: "Transcription failed",
  skipped: "Not transcribed",
});

export function speechSegmentStatusLabel(status) {
  return SEGMENT_STATUS_LABELS[String(status || "")] || String(status || "");
}

/** Why a segment has no transcript, in operator terms. */
export function speechSegmentReasonLabel(reason) {
  const text = String(reason || "").trim();
  if (!text) return "";
  if (text === "too_short") return "too short to transcribe";
  if (text === "queue_full") return "transcription queue was full";
  if (text === "shutdown" || text === "restart") return "SurvNG restarted before transcription";
  if (text === "timeout") return "transcription timed out";
  if (text === "worker_exited") return "transcriber stopped unexpectedly";
  if (text.includes("model not installed")) return "Whisper model not installed";
  if (text.includes("openvino-genai")) return "transcription runtime not installed";
  return "";
}

/** Short summary of the speech counts attached to an incident payload. */
export function incidentSpeechLabel(speech) {
  const count = Number(speech?.segments) || 0;
  if (!count) return "";
  const seconds = Math.round(Number(speech.seconds) || 0);
  const spoken = `${count} speech segment${count === 1 ? "" : "s"}${seconds ? ` · ${seconds}s` : ""}`;
  const transcribed = Number(speech.transcribed) || 0;
  return transcribed ? `${spoken} · ${transcribed} transcribed` : spoken;
}

/** Query string for GET /api/speech/segments. */
export function speechSegmentsQuery({ query = "", cameraId = "", limit = 100, beforeId = null } = {}) {
  const params = new URLSearchParams();
  const text = String(query || "").trim();
  if (text) params.set("q", text.slice(0, 200));
  if (cameraId) params.set("camera_id", String(cameraId));
  params.set("limit", String(Math.max(1, Math.min(500, Number(limit) || 100))));
  if (Number.isInteger(beforeId) && beforeId > 0) params.set("before_id", String(beforeId));
  return params.toString();
}

/** Recording context for jumping to the start of a segment on the timeline. */
export function speechSegmentRecordingContext(segment) {
  if (!segment?.camera_id || !segment?.started_at) return null;
  const epoch = new Date(segment.started_at).getTime() / 1000;
  if (!Number.isFinite(epoch)) return null;
  return { cameraId: segment.camera_id, epoch };
}

export function speechSearchMode(value) {
  return value === "speech" ? "speech" : "visual";
}
