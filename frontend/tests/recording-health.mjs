import assert from "node:assert/strict";
import { recordingHealth, recordingHealthContext } from "../src/recordingHealth.mjs";

const now = 1_000_000;
const base = (overrides = {}) => ({
  camerasUpdatedAt: now, systemUpdatedAt: now,
  appConfig: { cameras: [], retention: { minimum_free_percent: 15, emergency_free_percent: 5 } },
  cameras: [], system: { storage: { available: true, free_bytes: 20, total_bytes: 100, sampled_at: "2026-01-01T00:00:00Z" }, uptime_seconds: 3600 }, now,
  ...overrides,
});

let result = recordingHealth(base({
  appConfig: { cameras: [{ id: "sub", record: false, record_sub: true }] },
  cameras: [{ id: "sub", sub_recording: true, recording: false }],
}));
assert.equal(result.expectedCount, 1, "sub-only cameras count as expected");
assert.equal(result.activeCount, 1, "active substream satisfies a sub-only camera");

result = recordingHealth(base({ appConfig: { cameras: [{ id: "main", record: true, record_sub: false }] }, cameras: [{ id: "main", recording: false }] }));
assert.equal(result.rows[0].state, "attention");
assert.deepEqual(result.rows[0].missing, ["main stream"]);

result = recordingHealth(base({ appConfig: { cameras: [{ id: "stopped", record: true }] }, cameras: [{ id: "stopped", recording: false, recording_enabled: false }] }));
assert.equal(result.expectedCount, 0, "an explicitly stopped recorder is not expected to be active");
assert.equal(result.rows[0].state, "paused");

result = recordingHealth(base({ appConfig: { cameras: [{ id: "offline", record: true }] }, cameras: [{ id: "offline", recording: true, running: false, capture_connectivity: "offline" }] }));
assert.deepEqual(result.rows[0].missing, ["camera process", "capture"]);

result = recordingHealth(base({ appConfig: { cameras: [{ id: "off", enabled: false, record: true }] }, cameras: [{ id: "off", recording: false }] }));
assert.equal(result.rows[0].state, "disabled");
assert.equal(result.expectedCount, 0, "disabled cameras are not expected to record");

result = recordingHealth(base({ appConfig: { cameras: [{ id: "both", record: true, record_sub: true }] }, cameras: [{ id: "both", recording: true, sub_recording: false }] }));
assert.deepEqual(result.rows[0].missing, ["substream"]);

result = recordingHealth(base({ appConfig: null, cameras: null, system: null, camerasUpdatedAt: null, systemUpdatedAt: null }));
assert.equal(result.cameraFresh, false); assert.equal(result.storage.state, "unavailable"); assert.ok(result.issues > 0);

result = recordingHealth(base({ appConfig: null }));
assert.equal(result.cameraFresh, true); assert.equal(result.cameraDataKnown, false);
assert.equal(result.activeCount, null, "missing configuration cannot be represented as a healthy 0/0");

result = recordingHealth(base({ camerasError: true, systemError: true }));
assert.equal(result.cameraFresh, false); assert.equal(result.systemFresh, false);

result = recordingHealth(base({ camerasUpdatedAt: now - 90_001, systemUpdatedAt: now - 90_001,
  appConfig: { cameras: [{ id: "stale", record: true }] }, cameras: [{ id: "stale", recording: true }] }));
assert.equal(result.rows[0].state, "stale", "old successful snapshots must not stay healthy");
assert.equal(result.activeCount, null);
assert.equal(result.storage.state, "unavailable");
assert.equal(result.issues, 2);

result = recordingHealth(base({ system: { storage: { available: true, free_bytes: 0, total_bytes: 100 } } }));
assert.equal(result.storage.state, "critical", "zero space is an emergency, never unavailable or healthy");

result = recordingHealth(base({ system: { storage: { available: false, free_bytes: 0, total_bytes: 100 } } }));
assert.equal(result.storage.state, "unavailable");

result = recordingHealth(base({ system: { storage: { available: true, free_bytes: 0, total_bytes: 0, used_percent: 0 } } }));
assert.equal(result.storage.state, "unavailable", "invalid totals cannot fall back to a healthy percentage");

result = recordingHealth(base({ system: { storage: { available: true, free_bytes: 10, total_bytes: 100 } } }));
assert.equal(result.storage.state, "warning");
assert.match(recordingHealthContext(result).attention[0], /10.0% free.*15% cleanup threshold/);
result = recordingHealth(base({ system: { storage: { available: true, free_bytes: 3, total_bytes: 100 } } }));
assert.match(recordingHealthContext(result).attention[0], /3.0% free.*5% emergency threshold/);
result = recordingHealth(base({ appConfig: { cameras: [{ id: "gate", name: "Side gate", record: true, record_sub: true }] }, cameras: [{ id: "gate", recording: false, sub_recording: true }] }));
assert.deepEqual(recordingHealthContext(result).attention, ["Side gate: missing main stream."]);
assert.match(recordingHealthContext(result).recording[0], /0 of 1 expected cameras/);
result = recordingHealth(base({ camerasUpdatedAt: now - 90_001, systemUpdatedAt: now - 90_001 }));
assert.equal(recordingHealthContext(result).attention.length, 2);
assert.ok(recordingHealthContext(result).attention.every((reason) => reason.includes("unavailable or stale")));
assert.ok(!recordingHealthContext(result).storage[0].includes("20.0%"), "stale storage must not be described as current");
result = recordingHealth(base({ system: { ...base().system, lifecycle: "starting", detector: { enabled: true, loaded_backend: null } } }));
assert.equal(result.issues, 1, "system health remains visible on every workspace");
assert.deepEqual(recordingHealthContext(result).attention, ["System is starting.", "Detection is enabled but its backend is not loaded."]);
result = recordingHealth(base({ appConfig: { cameras: [{ id: "paused", record: true }] }, cameras: [{ id: "paused", recording_enabled: false }] }));
assert.match(recordingHealthContext(result).recording[1], /1 paused or disabled camera is excluded/);
assert.deepEqual(recordingHealthContext(result).attention, []);
console.log("recording health tests passed");
