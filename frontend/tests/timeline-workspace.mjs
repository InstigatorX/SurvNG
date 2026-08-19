import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { expectedTimelineCameras, filteredTimelineCameras, normalizedTimelinePlaybackRate, parseTimelineView, timelineCompanionGrid, timelineEventMatchesFilter, timelineEvidenceWindow, timelineStageCameras, timelineStagePage, timelineViewport, TIMELINE_PLAYBACK_RATES } from "../src/timelineWorkspace.mjs";

assert.deepEqual(TIMELINE_PLAYBACK_RATES, [0.5, 1, 2, 4]);
assert.equal(normalizedTimelinePlaybackRate("2"), 2);
assert.equal(normalizedTimelinePlaybackRate(3), 1);

const cameras = [
  { id: "front-door", name: "Front Door" },
  { id: "garage", name: "Lower Garage" },
];
assert.deepEqual(filteredTimelineCameras(cameras, "front"), [cameras[0]]);
assert.deepEqual(filteredTimelineCameras(cameras, "GARAGE"), [cameras[1]]);
assert.equal(filteredTimelineCameras(cameras, "missing").length, 0);
assert.equal(filteredTimelineCameras(cameras, "").length, 2);
assert.deepEqual(timelineStageCameras(cameras, "garage", 2).map((camera) => camera.id), ["garage", "front-door"]);
assert.equal(timelineStageCameras(Array.from({ length: 12 }, (_, index) => ({ id: `camera-${index}` })), "camera-9").length, 7);
const routeCameras = [
  { id: "gate", name: "Gate" },
  { id: "lower-garage", name: "Lower Garage" },
  { id: "upper-garage", name: "Upper Garage" },
  { id: "front-door", name: "Front Door" },
];
const routes = [
  { from_camera: "gate", to_camera: "lower-garage", enabled: true },
  { from_camera: "upper-garage", to_camera: "gate", enabled: true, bidirectional: true },
  { from_camera: "front-door", to_camera: "gate", enabled: true, bidirectional: false },
  { from_camera: "gate", to_camera: "lower-garage", enabled: true },
  { from_camera: "gate", to_camera: "missing", enabled: true },
  { from_camera: "gate", to_camera: "front-door", enabled: false },
];
assert.deepEqual(expectedTimelineCameras(routeCameras, routes, "gate").map((camera) => camera.id), ["lower-garage", "upper-garage"]);
assert.deepEqual(expectedTimelineCameras(routeCameras, routes, "front-door").map((camera) => camera.id), ["gate"]);
assert.deepEqual(expectedTimelineCameras(routeCameras, routes, "lower-garage").map((camera) => camera.id), []);
assert.deepEqual(timelineCompanionGrid(0), { columns: 1, rows: 1 });
assert.deepEqual(timelineCompanionGrid(2), { columns: 1, rows: 2 });
assert.deepEqual(timelineCompanionGrid(3), { columns: 1, rows: 3 });
assert.deepEqual(timelineCompanionGrid(4), { columns: 2, rows: 2 });
assert.deepEqual(timelineCompanionGrid(6), { columns: 2, rows: 3 });
assert.deepEqual(timelineCompanionGrid(99), { columns: 2, rows: 3 });
assert.deepEqual(timelineStagePage(Array.from({ length: 15 }, (_, index) => ({ id: index })), 2), { cameras: [{ id: 14 }], page: 2, pages: 3 });
assert.equal(timelineEventMatchesFilter({ has_objects: true, labels: ["person"] }, "people"), true);
assert.equal(timelineEventMatchesFilter({ has_objects: true, labels: ["car"] }, "vehicles"), true);
assert.equal(timelineEventMatchesFilter({ has_objects: false, labels: [] }, "motion"), true);
const evidence = [0, 10, 20, 30].map((incident_epoch) => ({ incident_epoch }));
assert.deepEqual(timelineEvidenceWindow(evidence, 18, 2).map((event) => event.incident_epoch), [10, 20]);
assert.deepEqual(timelineViewport(0, 24 * 3600, 12 * 3600, 2), { startEpoch: 11 * 3600, endEpoch: 13 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 15 * 60, 2), { startEpoch: 0, endEpoch: 2 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 23.75 * 3600, 2), { startEpoch: 22 * 3600, endEpoch: 24 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 12 * 3600, 24), { startEpoch: 0, endEpoch: 24 * 3600 });
const filteredEvidence = [
  { id: 1, incident_epoch: 10, has_objects: true, labels: ["person"] },
  { id: 2, incident_epoch: 11, has_objects: true, labels: ["car"] },
  { id: 3, incident_epoch: 12, has_objects: false, labels: [] },
].filter((event) => timelineEventMatchesFilter(event, "people"));
assert.deepEqual(timelineEvidenceWindow(filteredEvidence, 10, 12).map((event) => event.id), [1]);
assert.deepEqual(parseTimelineView("?camera=gate&date=2026-08-15&source=main&at=123.5&event=44101&filter=vehicles&inspector=ai&window=4&objects=0&thumbs=0&speed=2", "2026-08-16"), {
  cameraId: "gate", date: "2026-08-15", source: "main", at: 123.5, eventFilter: "vehicles", eventId: 44101,
  inspector: "ai", windowHours: 4, lanes: { object: false, motion: true }, thumbnails: false, speed: 2,
});
const appSource = readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
const stylesSource = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const recordingsSource = appSource.slice(appSource.indexOf("function RecordingsPage"), appSource.indexOf("function exportStatusLabel"));
assert.match(recordingsSource, /const timelineInspectorTriggerRef = useRef\(null\)/);
assert.match(recordingsSource, /if \(samePlaybackScope\) playAt\(view\.at, false\)/);
assert.match(recordingsSource, /<a className="recordings-v2-selected-event-image"[^>]+\/incidents\?event_ids=/);
assert.doesNotMatch(recordingsSource, /View full incident/);
assert.match(stylesSource, /\.recordings-v2-incidents\s*\{[\s\S]*?grid-column:\s*1\s*\/\s*-1;[\s\S]*?grid-row:\s*3;/);
assert.match(stylesSource, /\.recordings-v2-selected-event-image img\s*\{[^}]*object-fit:\s*contain;/);

console.log("timeline workspace tests passed");
