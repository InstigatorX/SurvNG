import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { expectedTimelineCameras, filteredTimelineCameras, invalidateTimelineIdentityCache, mergeTimelineIncidentIdentity, normalizedTimelinePlaybackRate, parseTimelineView, resolveTimelineHeroCameraId, timelineCompanionGrid, timelineEventMatchesFilter, timelineEvidenceWindow, timelineIdentityDetailEventId, timelineIncidentIncludesEvent, timelinePanViewport, timelinePlayheadInComfortZone, timelineStageCameras, timelineStagePage, timelineTickIntervalSeconds, timelineViewport, timelineViewportPage, TIMELINE_PLAYBACK_RATES } from "../src/timelineWorkspace.mjs";

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
assert.equal(timelineIdentityDetailEventId({ id: "incident:4", representative_event_id: 41 }), 41);
assert.equal(timelineIdentityDetailEventId({ id: "incident:4", events: [{ id: 42 }] }), 42);
assert.equal(timelineIdentityDetailEventId({ id: "not-an-event" }), null);
assert.equal(timelineIncidentIncludesEvent({ representative_event_id: 41, events: [{ id: 40 }] }, 41), true);
assert.equal(timelineIncidentIncludesEvent({ representative_event_id: 41, events: [{ id: 40 }] }, "40"), true);
assert.equal(timelineIncidentIncludesEvent({ representative_event_id: 41, events: [{ id: 40 }] }, 99), false);
const identityCache = new Map([
  [41, { representative_event_id: 41, events: [{ id: 40 }] }],
  [50, { representative_event_id: 50, events: [{ id: 49 }] }],
]);
assert.deepEqual(invalidateTimelineIdentityCache(identityCache, 40), [41]);
assert.deepEqual([...identityCache.keys()], [50]);
const timelineSummary = { id: "incident:4", incident_epoch: 10, labels: ["person"] };
assert.deepEqual(mergeTimelineIncidentIdentity(timelineSummary, {
  identities: [{ name: "Ada", status: "confirmed" }],
  primary_identity: { name: "Ada", status: "confirmed" },
  objects: [{ label: "different-detail-shape" }],
}), {
  ...timelineSummary,
  identities: [{ name: "Ada", status: "confirmed" }],
  primary_identity: { name: "Ada", status: "confirmed" },
});
assert.equal(
  mergeTimelineIncidentIdentity({ id: 42, incident_epoch: 10 }, { identities: [] }).snapshot_path,
  "available",
);
assert.equal(
  mergeTimelineIncidentIdentity(
    { id: 42, incident_epoch: 10 },
    { events: [{ id: 42, snapshot_path: "available" }] },
  ).snapshot_path,
  "available",
);
assert.deepEqual(timelineViewport(0, 24 * 3600, 12 * 3600, 2), { startEpoch: 11 * 3600, endEpoch: 13 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 15 * 60, 2), { startEpoch: 0, endEpoch: 2 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 23.75 * 3600, 2), { startEpoch: 22 * 3600, endEpoch: 24 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 12 * 3600, 24), { startEpoch: 0, endEpoch: 24 * 3600 });
assert.deepEqual(timelineViewport(0, 24 * 3600, 12 * 3600, 1), { startEpoch: 11.5 * 3600, endEpoch: 12.5 * 3600 });
const pannedBack = timelinePanViewport(0, 24 * 3600, { startEpoch: 11 * 3600, endEpoch: 13 * 3600 }, -1800);
assert.deepEqual(pannedBack, { startEpoch: 10.5 * 3600, endEpoch: 12.5 * 3600 });
assert.deepEqual(timelinePanViewport(0, 24 * 3600, pannedBack, 1800), { startEpoch: 11 * 3600, endEpoch: 13 * 3600 });
assert.deepEqual(timelinePanViewport(0, 24 * 3600, { startEpoch: 0, endEpoch: 2 * 3600 }, -3600), { startEpoch: 0, endEpoch: 2 * 3600 });
assert.deepEqual(timelinePanViewport(0, 24 * 3600, { startEpoch: 22 * 3600, endEpoch: 24 * 3600 }, 3600), { startEpoch: 22 * 3600, endEpoch: 24 * 3600 });
assert.equal(pannedBack.endEpoch - pannedBack.startEpoch, 2 * 3600);
const playhead = 12 * 3600;
assert.deepEqual(timelinePanViewport(0, 24 * 3600, { startEpoch: 11 * 3600, endEpoch: 13 * 3600 }, -3600).startEpoch, 10 * 3600);
assert.equal(playhead, 12 * 3600);
assert.equal(timelinePlayheadInComfortZone({ startEpoch: 11 * 3600, endEpoch: 13 * 3600 }, 12 * 3600), true);
assert.equal(timelinePlayheadInComfortZone({ startEpoch: 11 * 3600, endEpoch: 13 * 3600 }, 11.1 * 3600), false);
assert.deepEqual(timelineViewportPage(0, 24 * 3600, { startEpoch: 11 * 3600, endEpoch: 13 * 3600 }, 1), { startEpoch: 12 * 3600, endEpoch: 14 * 3600 });
assert.deepEqual(timelineViewportPage(0, 24 * 3600, { startEpoch: 0, endEpoch: 2 * 3600 }, -1), { startEpoch: 0, endEpoch: 2 * 3600 });
assert.deepEqual(timelineViewportPage(0, 24 * 3600, { startEpoch: 22 * 3600, endEpoch: 24 * 3600 }, 1), { startEpoch: 22 * 3600, endEpoch: 24 * 3600 });
assert.equal(resolveTimelineHeroCameraId(routeCameras, "all"), "gate");
assert.equal(resolveTimelineHeroCameraId(routeCameras, "front-door"), "front-door");
assert.equal(resolveTimelineHeroCameraId(routeCameras, "missing"), "gate");
assert.equal(resolveTimelineHeroCameraId([], "all"), "");
assert.equal(timelineTickIntervalSeconds(1), 5 * 60);
assert.equal(timelineTickIntervalSeconds(24), 3600);
assert.deepEqual(expectedTimelineCameras(routeCameras, routes, resolveTimelineHeroCameraId(routeCameras, "gate")).map((camera) => camera.id), ["lower-garage", "upper-garage"]);
const promotedHero = resolveTimelineHeroCameraId(routeCameras, "lower-garage");
assert.equal(promotedHero, "lower-garage");
assert.deepEqual(expectedTimelineCameras(routeCameras, routes, promotedHero).map((camera) => camera.id), []);
assert.equal(playhead, 12 * 3600);
const filteredEvidence = [
  { id: 1, incident_epoch: 10, has_objects: true, labels: ["person"] },
  { id: 2, incident_epoch: 11, has_objects: true, labels: ["car"] },
  { id: 3, incident_epoch: 12, has_objects: false, labels: [] },
].filter((event) => timelineEventMatchesFilter(event, "people"));
assert.deepEqual(timelineEvidenceWindow(filteredEvidence, 10, 12).map((event) => event.id), [1]);
assert.deepEqual(parseTimelineView("?camera=gate&date=2026-08-15&source=main&at=123.5&event=44101&filter=vehicles&inspector=ai&window=4&objects=0&thumbs=0&speed=2", "2026-08-16"), {
  cameraId: "gate", date: "2026-08-15", source: "main", at: 123.5, eventFilter: "vehicles", eventId: 44101,
  trailEventIds: [], trailQueryMode: null,
  inspector: "ai", windowHours: 4, lanes: { object: false, motion: true }, thumbnails: false, speed: 2,
});
assert.deepEqual(parseTimelineView("?camera=all&date=2026-08-15&at=123.5&window=1", "2026-08-16"), {
  cameraId: "all", date: "2026-08-15", source: null, at: 123.5, eventFilter: "all", eventId: null,
  trailEventIds: [], trailQueryMode: null,
  inspector: "details", windowHours: 1, lanes: { object: true, motion: true }, thumbnails: true, speed: 1,
});
assert.deepEqual(parseTimelineView("?camera=gate&event=84&trail=12,84,91&query_mode=visual", "2026-08-16").trailEventIds, [12, 84, 91]);
assert.equal(parseTimelineView("?camera=gate&event=84&trail=12,84,91&query_mode=visual", "2026-08-16").trailQueryMode, "visual");
assert.equal(resolveTimelineHeroCameraId(routeCameras, parseTimelineView("?camera=all", "2026-08-16").cameraId), "gate");
const appSource = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
const stylesSource = [
  readFileSync(new URL("../src/styles.css", import.meta.url), "utf8"),
  readFileSync(new URL("../src/timeline/timeline.css", import.meta.url), "utf8"),
  readFileSync(new URL("../src/shell/mobile.css", import.meta.url), "utf8"),
].join("\n");
const recordingsSource = appSource.slice(appSource.indexOf("function RecordingsPage"), appSource.indexOf("function exportStatusLabel"));
assert.match(recordingsSource, /recordings-v2-forensic-context/);
assert.match(recordingsSource, /forensicNav\.label/);
assert.match(recordingsSource, /stepForensicNav/);
assert.doesNotMatch(recordingsSource, /onClick=\{closeFrameSearch\}>Nearby/);
assert.doesNotMatch(recordingsSource, /recordings-event-inspector/);
assert.match(recordingsSource, /heroMuted/);
assert.match(recordingsSource, /if \(samePlaybackScope\) playAt\(view\.at, false\)/);
assert.match(appSource, /timeline-camera-picker/);
assert.match(recordingsSource, /<TimelineCameraPicker/);
assert.match(recordingsSource, /MobileCameraSelect/);
assert.match(recordingsSource, /timeline-mobile-camera-select/);
assert.match(appSource, /search-mobile-camera-select/);
assert.match(stylesSource, /\.timeline-camera-picker,\s*\.search-camera-list/);
assert.match(stylesSource, /\.timeline-mobile-camera-select/);
assert.match(stylesSource, /\.search-mobile-camera-select/);
assert.match(recordingsSource, /recordings-toolbar-day-controls/);
assert.match(recordingsSource, /recordings-commandbar-day-controls/);
assert.match(recordingsSource, /<TimelineDatePicker/);
assert.doesNotMatch(recordingsSource, /type="date"/);
assert.match(appSource, /allOption=\{\{ value: "", label: "All cameras" \}\}/);
assert.doesNotMatch(appSource, /<RecordingCameraRail/);
assert.match(recordingsSource, /onPanViewport=/);
assert.doesNotMatch(recordingsSource, /onPageViewport=/);
assert.match(recordingsSource, /investigation-hidden/);
assert.match(recordingsSource, /recordings-investigation-toggle/);
assert.match(recordingsSource, /selectedIncidentIdentityCacheRef/);
assert.match(recordingsSource, /frameSearchRequestRef = useRef\(null\)/);
assert.match(recordingsSource, /frameSearchRequestRef\.current\?\.abort\(\)/);
assert.match(recordingsSource, /clearVisualSearchTrail\(window\.sessionStorage\)/);
assert.match(recordingsSource, /signal: request/);
assert.match(recordingsSource, /frameSearchRequestRef\.current !== controller/);
assert.match(recordingsSource, /\/api\/incidents\/by-event\//);
assert.match(recordingsSource, /type !== "identity_update"/);
assert.match(recordingsSource, /selectedIdentityRevision\]\);/);
assert.match(recordingsSource, /recordings-return-playhead/);
assert.match(recordingsSource, /recordings-v2-scale/);
assert.match(recordingsSource, /shouldResumePlaybackAfterSeek/);
assert.match(recordingsSource, /seekVideoToTime/);
assert.match(recordingsSource, /recording-hero-controls/);
assert.match(recordingsSource, /function skipPlayback\(seconds, playing = autoplayRef\.current \|\| heroPlaying\)/);
assert.match(recordingsSource, /skipPlayback\(-10\)/);
assert.match(recordingsSource, /skipPlayback\(10\)/);
assert.match(appSource, /prefersJpegScrubPreview/);
assert.match(appSource, /scrubPreviewDelayMs/);
assert.match(recordingsSource, /completePendingRecordingSeek/);
assert.match(recordingsSource, /scheduleSeekWatchdog/);
assert.match(recordingsSource, /ignorePauseUntilRef/);
assert.match(stylesSource, /\.recording-hero-controls/);
assert.doesNotMatch(recordingsSource, /setEventFilter\("object"\)/);
assert.doesNotMatch(recordingsSource, /Thumbnails/);
assert.doesNotMatch(recordingsSource, /recordings-timeline-evidence/);
assert.doesNotMatch(recordingsSource, /recordings-commandbar-export/);
assert.doesNotMatch(recordingsSource, /View full incident/);
assert.match(stylesSource, /\.recordings-v2-incidents\s*\{[\s\S]*?grid-column:\s*1\s*\/\s*-1;/);
assert.match(stylesSource, /\.recordings-v2-selected-event-image img\s*\{[^}]*object-fit:\s*contain;/);
assert.match(stylesSource, /--timeline-companion-width:\s*clamp\(148px, 22vw, 280px\)/);
assert.doesNotMatch(recordingsSource.slice(recordingsSource.indexOf("return ("), recordingsSource.indexOf("<RecordingTimeline")), /recordings-v2-cameras/);

console.log("timeline workspace tests passed");
