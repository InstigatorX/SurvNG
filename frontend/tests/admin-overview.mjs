import assert from "node:assert/strict";
import {
  formatAgeShort,
  formatOverviewBytes,
  overviewAttention,
  overviewCameraRows,
  pendingReviewSuggestions,
  uptimePercent,
  uptimeSlots,
} from "../src/adminOverview.mjs";
import { recordingHealth } from "../src/recordingHealth.mjs";

const now = Date.now();
const appConfig = {
  retention: { minimum_free_percent: 15, emergency_free_percent: 5 },
  cameras: [
    { id: "front", name: "Front Door", record: true, homekit: { enabled: true } },
    { id: "garage", name: "Garage", record: true },
    { id: "shed", name: "Shed", record: false },
  ],
};
const runtime = [
  {
    id: "front", running: true, connected: true, capture_connectivity: "healthy", recording: true,
    stream_dimensions: { main: { width: 2560, height: 1440 } }, capture_stats: { main: { fps: 15.02 } },
  },
  {
    id: "garage", running: true, connected: false, capture_connectivity: "offline", recording: false,
    last_frame_age_seconds: 840, capture_reconnects: 6,
  },
  { id: "shed", running: true, connected: true, recording: false, detection_enabled: false },
];
const lowStorage = { available: true, free_bytes: 61 * 1024 ** 3, total_bytes: 2 * 1024 ** 4 };
const health = recordingHealth({
  cameras: runtime, appConfig, system: { storage: lowStorage }, camerasUpdatedAt: now, systemUpdatedAt: now, now,
});

// Attention queue: critical items first, then warnings, then information.
const attention = overviewAttention({
  health,
  detector: { enabled: true, loaded_backend: "" },
  retention: { state: "error", error: "Disk is read-only" },
  reviewSuggestions: 4,
  interruptions: { unexpected: 2 },
});
assert.deepEqual(attention.map((item) => item.id), ["camera-garage", "storage", "detector", "retention", "restarts", "review"]);
assert.equal(attention[0].title, "Garage is not recording");
assert.equal(attention[0].detail, "Missing main stream and capture. Last frame 14 min ago. 6 reconnects since restart.");
assert.deepEqual(attention[0].actions.map((action) => [action.id, action.cameraId, Boolean(action.primary)]), [
  ["diagnostics", "garage", false],
  ["open-camera", "garage", true],
]);
assert.equal(attention[1].severity, "critical");
assert.match(attention[1].detail, /^61\.0 GB free \(3\.0%\)\. Retention removes the oldest recordings below 5% free\.$/);
assert.equal(attention[4].title, "SurvNG restarted unexpectedly 2 times in the last 24 h");
assert.equal(attention[5].title, "System review has 4 suggestions");

// Healthy system: nothing to show, and a stopped-on-purpose camera is not an issue.
const healthy = recordingHealth({
  cameras: runtime.map((camera) => ({ ...camera, connected: true, capture_connectivity: "healthy", recording: true })),
  appConfig,
  system: { storage: { available: true, free_bytes: 1024 ** 4, total_bytes: 2 * 1024 ** 4 } },
  camerasUpdatedAt: now,
  systemUpdatedAt: now,
  now,
});
assert.deepEqual(overviewAttention({ health: healthy, detector: { enabled: true, loaded_backend: "openvino" } }), []);
assert.deepEqual(overviewAttention({ health: healthy, detector: { enabled: false } }), []);

// Stale camera data is reported once instead of blaming every camera.
const stale = recordingHealth({ cameras: runtime, appConfig, system: {}, camerasUpdatedAt: now - 600_000, now });
assert.deepEqual(overviewAttention({ health: stale }).map((item) => item.id), ["camera-status"]);

// Camera table.
const availability = {
  front: [{ started_at: "a", availability_percent: 100 }, { started_at: "b", availability_percent: 100 }],
  garage: [{ started_at: "a", availability_percent: 100 }, { started_at: "b", availability_percent: 20 }],
};
const rows = overviewCameraRows({ health, cameras: appConfig.cameras, availability });
assert.deepEqual(rows.map((row) => [row.name, row.status.label, row.status.tone]), [
  ["Front Door", "Recording", "good"],
  ["Garage", "Offline 14 min", "bad"],
  ["Shed", "Not recording by choice", "muted"],
]);
assert.equal(rows[0].stream, "2560×1440 · 15 fps");
assert.equal(rows[1].stream, "—");
assert.deepEqual(rows.map((row) => [row.detection, row.homekit]), [["On", "On"], ["On", "Off"], ["Off", "Off"]]);
assert.deepEqual(rows[1].uptime.map((slot) => slot.tone), ["good", "offline"]);
assert.equal(rows[1].uptimePercent, 60);
assert.deepEqual(rows[2].uptime, []);

// Helpers.
assert.deepEqual(uptimeSlots([{ availability_percent: null }, { availability_percent: 99.5 }, { availability_percent: 75 }]).map((slot) => [slot.tone, slot.label]), [
  ["none", "No data"],
  ["good", "99.5% available"],
  ["degraded", "75.0% available"],
]);
assert.equal(uptimePercent([{ availability_percent: null }]), null);
assert.equal(pendingReviewSuggestions({ briefing: { result: { recommendations: [{}, { dismissed: true }, { applied: true }, {}] } } }), 2);
assert.equal(pendingReviewSuggestions(null), 0);
assert.equal(formatOverviewBytes(1.42 * 1024 ** 4), "1.42 TB");
assert.equal(formatOverviewBytes(undefined), "—");
assert.equal(formatAgeShort(30), "30 s");
assert.equal(formatAgeShort(7200), "2 h");

console.log("admin overview tests passed");
