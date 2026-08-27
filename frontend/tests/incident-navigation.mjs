import assert from "node:assert/strict";
import { adjacentIncident, createIncidentPageCache, incidentArrowNavigationAllowed, incidentDetectionFrameSize, incidentDetailQuery, incidentEvidenceFrames, incidentIndexForEvent, incidentMosaicEvents, incidentMosaicPage, incidentObjectFocusCropRect, incidentObjectFocusMaxScale, incidentObjectFocusStyle, incidentObjectFocusThumbnailWidth, incidentObjectIconName, incidentProgressiveImageWidth, incidentSelectionHref, incidentThumbnailObjectFocusEnabled, incidentThumbnailPageSize, incidentTrackingFrameSize, incidentZoomLayout, incidentsNewestFirst, incidentTriggerLabel, linkedIncidentEventFilter, normalizeIncidentThumbnailObjectFocus, normalizeIncidentThumbnailObjectFocusZoom, retainFocusedIncident, showIncidentCardAnnotations } from "../src/incidentNavigation.mjs";

const incidents = [
  { id: 100, events: [{ id: 101 }, { id: 102 }] },
  { id: 200, events: [{ id: 201 }] },
  { id: "300", events: [{ id: 301 }] },
];

assert.equal(incidentIndexForEvent(incidents, incidents[0]), 0);
assert.equal(incidentIndexForEvent(incidents, { id: 102 }), 0);
assert.equal(incidentIndexForEvent(incidents, { id: 999 }), -1);
assert.equal(adjacentIncident(incidents, incidents[0], 1), incidents[1]);
assert.equal(adjacentIncident(incidents, { id: 101 }, 1), incidents[1]);
assert.equal(adjacentIncident(incidents, { id: 201 }, -1), incidents[0]);
assert.equal(adjacentIncident(incidents, { id: 301 }, 1), incidents[0]);
assert.equal(adjacentIncident([incidents[0]], incidents[0], 1), null);
assert.equal(incidentArrowNavigationAllowed({ closest: () => null }), true);
assert.equal(incidentArrowNavigationAllowed({ closest: () => ({ tagName: "VIDEO" }) }), false);
assert.equal(incidentSelectionHref("https://example.test/survng/incidents?day=2026-08-16", 42, "/survng"), "/survng/incidents?day=2026-08-16&event_ids=42");
assert.equal(incidentSelectionHref("https://example.test/survng/incidents?event_ids=1", 42, "/survng"), "/survng/incidents?event_ids=42");
assert.equal(incidentSelectionHref("/survng/incidents", "bad", "/survng"), "/survng/incidents");
assert.equal(incidentSelectionHref("/incidents?day=2026-08-16", 42, "/survng"), "/survng/incidents?day=2026-08-16&event_ids=42");
assert.equal(incidentSelectionHref("/incidents?event_ids=1", 42), "/incidents?event_ids=42");
assert.equal(incidentSelectionHref("/incidents#focus", 7), "/incidents?event_ids=7#focus");
assert.equal(incidentDetailQuery(incidents[0]), "event_ids=101%2C102&gap_seconds=45");
assert.equal(incidentDetailQuery({ events: [{ id: 101 }, { id: 101 }, { id: "bad" }] }), "event_ids=101&gap_seconds=45");
assert.equal(incidentDetailQuery({ events: [] }), "");
assert.equal(linkedIncidentEventFilter({ kind: "motion", object_event_count: 0 }), "motion");
assert.equal(linkedIncidentEventFilter({ kind: "motion", object_event_count: 1 }), "object");
assert.equal(linkedIncidentEventFilter({ kind: "object" }), "object");

const mosaicEvents = incidentMosaicEvents({ events: [
  { id: 3, created_at: "2026-08-08T12:00:03Z" },
  { id: 1, created_at: "2026-08-08T12:00:01Z" },
  { id: 2, created_at: "2026-08-08T12:00:02Z" },
  { id: 4, created_at: "invalid" },
] });
assert.deepEqual(mosaicEvents.map((event) => event.id), [1, 2, 3, 4]);
assert.deepEqual(incidentMosaicEvents({}), []);
assert.deepEqual(incidentMosaicPage(mosaicEvents, 0, 2), { items: mosaicEvents.slice(0, 2), page: 0, pageCount: 2 });
assert.deepEqual(incidentMosaicPage(mosaicEvents, 99, 2), { items: mosaicEvents.slice(2), page: 1, pageCount: 2 });
assert.deepEqual(incidentMosaicPage([], 0), { items: [], page: 0, pageCount: 1 });
const evidenceFrames = incidentEvidenceFrames({
  id: 10,
  created_at: "2026-08-08T12:00:00Z",
  objects: [{
    label: "person",
    confidence: 0.8,
    temporal_peak_confidence: 0.94,
    temporal_peak_confidence_offset_seconds: 4.5,
    temporal_sample_offset_seconds: 8,
  }],
  object_tracking: { tracks: [{ label: "person", box_history: [
    [1775649601, 0, 0, 10, 10],
    [1775649603, 0, 0, 20, 20],
  ] }] },
});
assert.deepEqual(evidenceFrames.map(({ key, label, epoch, kind }) => ({ key, label, epoch, kind })), [
  { key: "trigger", label: "Trigger", epoch: 1786190400, kind: "recording" },
  { key: "detection", label: "Best person", epoch: 1786190404.5, kind: "recording" },
  { key: "selected", label: "Selected", epoch: 1786190408, kind: "snapshot" },
  { key: "tracking", label: "Best person track", epoch: 1775649603, kind: "recording" },
]);
assert.equal(incidentEvidenceFrames({
  created_epoch: 100,
  objects: [{ label: "car", temporal_sample_offset_seconds: 8 }],
})[1].label, "Detected car");
assert.deepEqual(incidentEvidenceFrames({}), []);

assert.equal(showIncidentCardAnnotations(false, true), true);
assert.equal(showIncidentCardAnnotations(false, false), false);
assert.equal(showIncidentCardAnnotations(true, true), false);
assert.equal(normalizeIncidentThumbnailObjectFocus("AUTO"), "auto");
assert.equal(normalizeIncidentThumbnailObjectFocus("button"), "button");
assert.equal(normalizeIncidentThumbnailObjectFocus("nope"), "off");
assert.equal(incidentThumbnailObjectFocusEnabled("auto"), true);
assert.equal(incidentThumbnailObjectFocusEnabled("off"), false);
assert.equal(normalizeIncidentThumbnailObjectFocusZoom(2.5), 2.5);
assert.equal(normalizeIncidentThumbnailObjectFocusZoom(0.5), 0.5);
assert.equal(normalizeIncidentThumbnailObjectFocusZoom(0.2), 0.25);
assert.equal(normalizeIncidentThumbnailObjectFocusZoom(9), 5.5);
assert.equal(incidentObjectFocusThumbnailWidth(140, 2, 1), 1280);
assert.equal(incidentObjectFocusThumbnailWidth(180, 2, 2), 1920);
assert.equal(incidentObjectFocusThumbnailWidth(220, 3, 3), 2560);
assert.ok(Math.abs(incidentObjectFocusMaxScale(720, 160, 2) - 3.0375) < 1e-9);
assert.ok(incidentObjectFocusMaxScale(2560, 160, 2) >= 5);
{
  const loose = incidentObjectFocusCropRect(2000, 400, [{ x1: 900, y1: 150, x2: 1100, y2: 250 }], 0.5);
  const fitted = incidentObjectFocusCropRect(2000, 400, [{ x1: 900, y1: 150, x2: 1100, y2: 250 }], 1);
  const tight = incidentObjectFocusCropRect(2000, 400, [{ x1: 900, y1: 150, x2: 1100, y2: 250 }], 2);
  assert.ok(loose && fitted && tight);
  assert.ok(loose.width * loose.height > fitted.width * fitted.height);
  assert.ok(fitted.width * fitted.height > tight.width * tight.height);
}
{
  const crop = incidentObjectFocusCropRect(
    1920,
    1080,
    [{ x1: 400, y1: 200, x2: 600, y2: 500 }],
    1,
    16,
    9,
  );
  assert.ok(crop);
  assert.ok(Math.abs((crop.width / crop.height) - (16 / 9)) < 0.02);
  assert.ok(crop.x1 <= 400 && crop.y1 <= 200 && crop.x2 >= 600 && crop.y2 >= 500);
}
{
  const base = incidentObjectFocusStyle(
    { width: 200, height: 100 },
    [{ left: 80, top: 40, width: 40, height: 20 }],
    1,
  );
  const tighter = incidentObjectFocusStyle(
    { width: 200, height: 100 },
    [{ left: 80, top: 40, width: 40, height: 20 }],
    2,
  );
  const looser = incidentObjectFocusStyle(
    { width: 200, height: 100 },
    [{ left: 80, top: 40, width: 40, height: 20 }],
    0.5,
  );
  const capped = incidentObjectFocusStyle(
    { width: 200, height: 100 },
    [{ left: 80, top: 40, width: 40, height: 20 }],
    5,
    1.5,
  );
  assert.ok(base);
  assert.ok(tighter);
  assert.ok(looser);
  const baseScale = Number(/scale\(([^)]+)\)/.exec(base.transform)?.[1]);
  const tighterScale = Number(/scale\(([^)]+)\)/.exec(tighter.transform)?.[1]);
  const looserScale = Number(/scale\(([^)]+)\)/.exec(looser.transform)?.[1]);
  const cappedScale = Number(/scale\(([^)]+)\)/.exec(capped.transform)?.[1]);
  assert.ok(baseScale >= 1);
  assert.ok(tighterScale > baseScale);
  assert.ok(looserScale < baseScale);
  assert.ok(looserScale >= 1);
  assert.equal(cappedScale, 1.5);
  assert.equal(incidentObjectFocusStyle({ width: 200, height: 100 }, [], 1), null);
}
assert.equal(incidentProgressiveImageWidth(0, 1), 1280);
assert.equal(incidentProgressiveImageWidth(390, 3), 1280);
assert.equal(incidentProgressiveImageWidth(1440, 1), 1920);
assert.equal(incidentProgressiveImageWidth(1600, 1.5), 2560);
assert.equal(incidentProgressiveImageWidth(1920, 2), 2560);
assert.equal(incidentZoomLayout({ width: 1000, height: 600 }, { scale: 1, x: 0, y: 0 }), null);
assert.deepEqual(
  incidentZoomLayout({ width: 1000, height: 600 }, { scale: 2, x: 40, y: -20 }),
  { left: -460, top: -320, width: 2000, height: 1200 },
);
assert.equal(incidentTriggerLabel({ trigger_source: "camera" }), "Camera");
assert.equal(incidentTriggerLabel({ trigger_source: "ema" }), "EMA");
assert.equal(incidentTriggerLabel({ trigger_source: "visual_backup" }), "EMA");
assert.equal(incidentTriggerLabel({}), "Camera");
assert.equal(incidentObjectIconName("person"), "person");
assert.equal(incidentObjectIconName("robot lawnmower"), "mower");
assert.equal(incidentObjectIconName("motorcycle"), "bike");
assert.equal(incidentObjectIconName("custom_class"), "object");
assert.deepEqual(incidentDetectionFrameSize({
  objects: [{ detection_frame_width: 2560, detection_frame_height: 1920 }],
}), { width: 2560, height: 1920 });
assert.deepEqual(incidentDetectionFrameSize({
  object_tracking: { frame_width: 1280, frame_height: 720 },
  objects: [{ detection_frame_width: 2560, detection_frame_height: 1920 }],
}), { width: 2560, height: 1920 });
assert.deepEqual(incidentTrackingFrameSize({
  object_tracking: { frame_width: 1280, frame_height: 720 },
  objects: [{ detection_frame_width: 2560, detection_frame_height: 1920 }],
}), { width: 1280, height: 720 });
assert.equal(incidentDetectionFrameSize({ objects: [{}] }), null);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 500, density: "compact" }), 6);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 720, density: "compact" }), 8);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 500, density: "comfortable" }), 4);
assert.equal(incidentThumbnailPageSize({ width: 400, height: 500, density: "compact", gap: 10, rowHeight: 90 }), 5);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 500, density: "compact", columns: 2, gap: 10, horizontalPadding: 24 }), 8);
assert.equal(incidentThumbnailPageSize({ width: 0, height: 0, density: "compact" }), 12);

const unorderedIncidents = [
  { id: "old", last_epoch: 100 },
  { id: "new", end_at: "2026-07-30T18:00:00Z" },
  { id: "middle", start_epoch: 200 },
  { id: "same-a", last_epoch: 150 },
  { id: "same-b", last_epoch: 150 },
];
const orderedIncidents = incidentsNewestFirst(unorderedIncidents);
assert.deepEqual(orderedIncidents.map((incident) => incident.id), ["new", "middle", "same-a", "same-b", "old"]);
assert.notEqual(orderedIncidents, unorderedIncidents);
assert.deepEqual(incidentsNewestFirst(null), []);
const retainedIncident = { id: 42, camera_id: "gate", detail: true };
assert.equal(retainFocusedIncident([{ id: "42", detail: false }], 42, retainedIncident)?.detail, false);
assert.equal(retainFocusedIncident([], "42", retainedIncident), retainedIncident);
assert.equal(retainFocusedIncident([], "99", retainedIncident), null);
assert.equal(retainFocusedIncident([], null, retainedIncident), null);

const loadedPages = [];
const pageCache = createIncidentPageCache(async (key) => {
  loadedPages.push(key);
  return { key };
});
const firstPending = pageCache.load("page-1");
assert.equal(pageCache.load("page-1"), firstPending);
assert.deepEqual(await firstPending, { key: "page-1" });
assert.deepEqual(pageCache.peek("page-1"), { key: "page-1" });
assert.deepEqual(loadedPages, ["page-1"]);
await pageCache.load("page-2");
await pageCache.load("page-3");
pageCache.retain(["page-2", "page-3"]);
assert.equal(pageCache.size(), 2);
await pageCache.load("page-1");
assert.deepEqual(loadedPages, ["page-1", "page-2", "page-3", "page-1"]);
pageCache.invalidate("page-1");
assert.equal(pageCache.peek("page-1"), undefined);
await pageCache.load("page-1");
assert.deepEqual(loadedPages, ["page-1", "page-2", "page-3", "page-1", "page-1"]);
pageCache.clear();
assert.equal(pageCache.size(), 0);
assert.equal(pageCache.peek("page-1"), undefined);

let retryAttempts = 0;
const retryingCache = createIncidentPageCache(async () => {
  retryAttempts += 1;
  if (retryAttempts === 1) throw new Error("temporary failure");
  return "recovered";
});
await assert.rejects(retryingCache.load("page"), /temporary failure/);
assert.equal(await retryingCache.load("page"), "recovered");
assert.equal(retryAttempts, 2);

console.log("incident navigation tests passed");
