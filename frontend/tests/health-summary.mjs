import assert from "node:assert/strict";
import {
  healthCameraRows,
  healthFindings,
  latestValue,
  memoryTrendBytesPerHour,
  seriesValues,
  trackingCapacitySummary,
  unavailableMinutes,
} from "../src/healthSummary.mjs";

const slots = (values, interruptions = []) => values.map((value, index) => ({
  started_at: `2026-11-21T${String(index).padStart(2, "0")}:00:00+00:00`,
  availability_percent: value,
  capture_interruptions: interruptions[index] || 0,
}));
const data = {
  cameras: [
    { id: "front", name: "Front Door", capture_reconnects: 0, activity: { last_24h: { events: 38 } } },
    { id: "garage", name: "Garage", capture_reconnects: 6, connected: false, capture_connectivity: "offline", activity: { last_24h: { events: 9 } } },
    { id: "gate", name: "Side Gate", capture_reconnects: 1, activity: { last_24h: { events: 11 } } },
  ],
  interruption_summary: { unexpected: 1 },
  detector: { runtime: { failed_inferences: 3 } },
  tracking_capacity_history: { short: [{ skipped: 1 }, { skipped: 2 }, { skipped: null }] },
};
const availability = {
  front: slots([100, 100, 100]),
  garage: slots([100, 100, 20], [0, 0, 6]),
  gate: slots([100, 100, 100], [1, 2, 1]),
};

// "now" is the end of the last slot so every slot counts as a full hour.
const END = new Date("2026-11-21T03:00:00+00:00").getTime();
const rows = healthCameraRows(data, availability, "", END);
assert.deepEqual(rows.map((row) => [row.name, Number(row.uptimePercent.toFixed(1)), row.unavailableMinutes, row.reconnects, row.interruptions, row.events]), [
  ["Front Door", 100, 0, 0, 0, 38],
  ["Garage", 73.3, 48, 6, 6, 9],
  ["Side Gate", 100, 0, 1, 4, 11],
]);
assert.deepEqual(healthCameraRows(data, availability, "gate").map((row) => row.id), ["gate"]);
assert.equal(healthCameraRows(data, {}, "")[0].uptimePercent, null);

const findings = healthFindings({ data, rows });
assert.deepEqual(findings.map((item) => [item.id, item.severity]), [
  ["uptime-garage", "critical"],
  ["interruptions-gate", "warning"],
  ["restarts", "warning"],
  ["detector-failures", "warning"],
  ["tracking-skipped", "warning"],
]);
assert.equal(findings[0].title, "Garage is offline now");
assert.equal(findings[0].detail.startsWith("73.3% available over 24 h (about 48 min unavailable) with 6 stream interruptions."), true);
// A past outage that recovered is a warning with its duration.
const recovered = healthFindings({ data: {}, rows: healthCameraRows({ cameras: [data.cameras[0]] }, { front: slots([100, 95, 100]) }, "", END) });
assert.deepEqual(recovered.map((item) => [item.severity, item.title]), [["warning", "Front Door was unavailable for about 3 min in the last 24 h"]]);
assert.equal(findings[1].title, "Side Gate: 4 stream interruptions in the last 24 h");
assert.equal(findings[4].title, "Object tracking skipped 3 sessions in the last 2 h");
assert.deepEqual(healthFindings({ data: {}, rows: healthCameraRows({ cameras: data.cameras.slice(0, 1) }, availability, "", END) }), []);
// A low current hour from an outage that already ended is not "offline now".
const reconnected = healthCameraRows({ cameras: [{ id: "garage", name: "Garage", connected: true }] }, { garage: slots([100, 100, 20]) }, "", END);
assert.equal(healthFindings({ data: {}, rows: reconnected })[0].title, "Garage was unavailable for about 48 min in the last 24 h");
// An offline camera without history is still reported.
assert.equal(healthFindings({ data: {}, rows: healthCameraRows({ cameras: [{ id: "x", name: "X", connected: false }] }, {}, "", END) })[0].title, "X is offline now");

assert.deepEqual(seriesValues([{ a: 1 }, { a: null }, {}, { a: 3 }], "a"), [1, 3]);
assert.equal(latestValue([{ a: 1 }, { a: 2 }, { a: null }], "a"), 2);
assert.equal(latestValue([], "a"), null);
assert.equal(unavailableMinutes([{ availability_percent: 50 }, { availability_percent: null }]), 30);
// The current hour only counts elapsed minutes: 5 minutes in, fully down = 5 minutes.
assert.equal(unavailableMinutes([{ started_at: "2026-11-21T12:00:00Z", availability_percent: 0 }], new Date("2026-11-21T12:05:00Z").getTime()), 5);
const memory = [
  { sampled_at: "2026-11-21T00:00:00Z", rss_bytes: 100 * 1024 ** 2 },
  { sampled_at: "2026-11-21T10:00:00Z", rss_bytes: 140 * 1024 ** 2 },
];
assert.equal(memoryTrendBytesPerHour(memory), 4 * 1024 ** 2);
assert.equal(memoryTrendBytesPerHour(memory.slice(0, 1)), null);

// Object tracking: share of requests that started without waiting.
{
  const summary = trackingCapacitySummary([
    { attempts: 10, waited: 0, skipped: 0, wait_seconds_max: 0 },
    // Skips waited first (limit above zero): late = waits.
    { attempts: 5, waited: 3, skipped: 1, wait_seconds_max: 5 },
    // Limit of zero: skips never wait, so late = skips.
    { attempts: 5, waited: 0, skipped: 2, wait_seconds_max: 0 },
  ]);
  assert.deepEqual(summary.lateSeries, [0, 3, 2]);
  assert.equal(summary.requests, 20);
  assert.equal(summary.waited, 3);
  assert.equal(summary.skipped, 3);
  assert.equal(summary.onTimePercent, 75);
  assert.equal(summary.longestWaitSeconds, 5);
  assert.equal(trackingCapacitySummary([{ attempts: 0 }]).onTimePercent, null);
  assert.equal(trackingCapacitySummary(null).requests, 0);
  // Bad counts never push a bucket past its own requests.
  assert.deepEqual(trackingCapacitySummary([{ attempts: 1, waited: 4, skipped: -2 }]).lateSeries, [1]);
}

console.log("health summary tests passed");
