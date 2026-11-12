import assert from "node:assert/strict";
import {
  CAMERA_SPEECH_DEFAULTS,
  cameraSpeechSettings,
  resolvedWhisperModel,
  speechFailureLabel,
  speechListenerLabel,
  speechSearchMode,
  speechSegmentReasonLabel,
  speechSegmentRecordingContext,
  speechSegmentStatusLabel,
  speechSegmentsQuery,
  speechSettings,
  withSetting,
} from "../src/speechSettings.mjs";

// Defaults keep speech off and fill fields missing from older configs.
assert.equal(speechSettings({}).enabled, false);
assert.equal(speechSettings({ speech: { enabled: true } }).transcription_device, "CPU");
assert.deepEqual(cameraSpeechSettings({ id: "gate" }), { ...CAMERA_SPEECH_DEFAULTS });
assert.equal(cameraSpeechSettings({ speech: { enabled: true, source: "live" } }).source, "live");
assert.deepEqual(withSetting({ a: 1 }, "b", 2), { a: 1, b: 2 });

// Automatic Whisper model follows the device; explicit choices win.
assert.equal(resolvedWhisperModel({ transcription_device: "CPU", transcription_model: "auto" }), "base.en");
assert.equal(resolvedWhisperModel({ transcription_device: "GPU", transcription_model: "auto" }), "small.en");
assert.equal(resolvedWhisperModel({ transcription_device: "GPU", transcription_model: "tiny.en" }), "tiny.en");
assert.equal(resolvedWhisperModel(undefined), "base.en");

// Labels are friendly and never leak unknown failure strings.
assert.equal(speechListenerLabel("no_audio_track"), "No audio track");
assert.equal(speechListenerLabel("mystery"), "Not running");
assert.equal(speechFailureLabel("stream_unavailable"), "Stream unavailable");
assert.equal(speechFailureLabel("rtsp://user:pw@host failed"), "");
assert.equal(speechSegmentStatusLabel("low_confidence"), "Low confidence");

// Segment queries are bounded and omit empty filters.
assert.equal(speechSegmentsQuery(), "limit=100");
assert.equal(
  speechSegmentsQuery({ query: "  front door ", cameraId: "gate", limit: 9999, beforeId: 42 }),
  "q=front+door&camera_id=gate&limit=500&before_id=42",
);
assert.equal(speechSegmentsQuery({ query: "x".repeat(300) }).length, "q=".length + 200 + "&limit=100".length);
assert.equal(speechSegmentsQuery({ beforeId: -1 }), "limit=100");

// Timeline links start at the segment's start time.
assert.deepEqual(
  speechSegmentRecordingContext({ camera_id: "gate", started_at: "2026-11-12T13:00:00+00:00" }),
  { cameraId: "gate", epoch: Date.parse("2026-11-12T13:00:00Z") / 1000 },
);
assert.equal(speechSegmentRecordingContext({ camera_id: "gate" }), null);
assert.equal(speechSegmentRecordingContext({ camera_id: "gate", started_at: "not a date" }), null);

assert.equal(speechSegmentReasonLabel("too_short"), "too short to transcribe");
assert.equal(speechSegmentReasonLabel("GPU: model not installed; CPU: model not installed"), "Whisper model not installed");
assert.equal(speechSegmentReasonLabel("SomeInternalError"), "");
assert.equal(speechSegmentReasonLabel(""), "");

assert.equal(speechSearchMode("speech"), "speech");
assert.equal(speechSearchMode("anything"), "visual");
assert.equal(speechSearchMode(null), "visual");

console.log("speech settings tests passed");

import { incidentSpeechLabel } from "../src/speechSettings.mjs";
assert.equal(incidentSpeechLabel(undefined), "");
assert.equal(incidentSpeechLabel({ segments: 0 }), "");
assert.equal(incidentSpeechLabel({ segments: 1, seconds: 3.4, transcribed: 0 }), "1 speech segment · 3s");
assert.equal(incidentSpeechLabel({ segments: 3, seconds: 20, transcribed: 2 }), "3 speech segments · 20s · 2 transcribed");
console.log("incident speech label tests passed");
