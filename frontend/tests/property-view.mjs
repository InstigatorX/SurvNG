import assert from "node:assert/strict";
import {
  aimFromPointer,
  applyDetection,
  conePolygon,
  configuredZoneRows,
  detectionIcon,
  highlightOpacity,
  HIGHLIGHT_FADE_MS,
  HIGHLIGHT_PEAK_OPACITY,
  moveZonePoint,
  overlayPresentation,
  panPropertyView,
  PROPERTY_ZOOM_MAX,
  propertyLocationFields,
  pruneHighlights,
  recordZonePointInsertion,
  rectangleZonePoints,
  removeZonePoint,
  rotationFromPointer,
  stackPropertyOverlays,
  translateZonePoints,
  undoZonePointInsertion,
  zoneCentroid,
  zoomPropertyView,
} from "../src/property/propertyView.mjs";

const now = Date.parse("2026-10-11T12:00:00Z");

assert.equal(highlightOpacity(now, now), HIGHLIGHT_PEAK_OPACITY);
assert.equal(highlightOpacity(now - HIGHLIGHT_FADE_MS, now), 0);
assert.equal(highlightOpacity(now - HIGHLIGHT_FADE_MS / 2, now), HIGHLIGHT_PEAK_OPACITY / 2);
assert.equal(highlightOpacity(now + 1000, now), HIGHLIGHT_PEAK_OPACITY);

assert.equal(detectionIcon("Person"), "person");
assert.equal(detectionIcon("truck"), "vehicle");
assert.equal(detectionIcon("dog"), "animal");
assert.equal(detectionIcon("bicycle"), "bicycle");
assert.equal(detectionIcon("package"), "package");
assert.equal(detectionIcon("unknown-label"), "generic");

const fresh = applyDetection({}, {
  camera_id: "gate",
  timestamp: "2026-10-11T12:00:00Z",
  zones: ["drive"],
  classes: ["person", "car"],
}, now);
assert.equal(fresh["gate\u0000drive"].classes.length, 2);
const older = applyDetection(fresh, {
  camera_id: "gate",
  timestamp: "2026-10-11T11:59:00Z",
  zones: ["drive"],
  classes: ["dog"],
}, now);
assert.deepEqual(older["gate\u0000drive"].classes, ["person", "car"]);
const ignored = applyDetection(fresh, {
  camera_id: "gate",
  timestamp: "2026-10-11T11:00:00Z",
  zones: ["yard"],
  classes: ["person"],
}, now);
assert.equal(ignored["gate\u0000yard"], undefined);
assert.deepEqual(pruneHighlights(fresh, now + HIGHLIGHT_FADE_MS), {});

const zones = configuredZoneRows({
  cameras: [{
    id: "gate",
    name: "Gate",
    zones: [{ name: "drive", color: "#22c55e", notifications_enabled: false }],
  }],
});
assert.equal(overlayPresentation({ camera_id: "gate", zone_name: "drive" }, zones).stale, false);
assert.equal(overlayPresentation({ camera_id: "gate", zone_name: "drive" }, zones).color, "#22c55e");
assert.equal(overlayPresentation({ camera_id: "gate", zone_name: "missing" }, zones).stale, true);

const cone = conePolygon({ x: 0.5, y: 0.5, heading_deg: 0, hfov_deg: 90, range_m: 12 }, 120, 120);
assert.ok(cone[0].y === 0.5);
assert.ok(cone[Math.floor(cone.length / 2)].y < 0.5);
assert.ok(Math.abs(cone[Math.floor(cone.length / 2)].x - 0.5) < 0.001);
const aim = aimFromPointer({ x: 0.5, y: 0.5 }, { x: 0.5, y: 0.3 }, 100, 100);
assert.ok(aim.heading_deg < 1 || aim.heading_deg > 359);
assert.equal(aim.range_m, 20);
const east = aimFromPointer({ x: 0.5, y: 0.5 }, { x: 0.7, y: 0.5 }, 100, 100);
assert.ok(Math.abs(east.heading_deg - 90) < 0.001);
assert.equal(rotationFromPointer({ x: 0.5, y: 0.5 }, { x: 0.7, y: 0.5 }), 90);

const frame = { width: 400, height: 400 };
const centered = zoomPropertyView({ scale: 1, x: 0, y: 0 }, frame, { x: 0, y: 0 }, 2);
assert.equal(centered.scale, 2);
assert.equal(centered.x, 0);
assert.equal(centered.y, 0);
const towardRight = zoomPropertyView({ scale: 1, x: 0, y: 0 }, frame, { x: 100, y: 0 }, 2);
assert.equal(towardRight.scale, 2);
assert.equal(towardRight.x, -100);
assert.equal(zoomPropertyView(towardRight, frame, { x: 100, y: 0 }, 0.5).scale, 1);
assert.deepEqual(zoomPropertyView({ scale: 1, x: 40, y: 20 }, frame, { x: 0, y: 0 }, 0.5), { scale: 1, x: 0, y: 0 });
assert.equal(zoomPropertyView({ scale: 1, x: 0, y: 0 }, frame, { x: 0, y: 0 }, 100).scale, PROPERTY_ZOOM_MAX);
const shoved = panPropertyView({ scale: 2, x: 0, y: 0 }, frame, -1000, 40);
assert.equal(shoved.x, -200);
assert.equal(shoved.y, 40);
assert.deepEqual(panPropertyView({ scale: 1, x: 0, y: 0 }, frame, 30, 30), { scale: 1, x: 0, y: 0 });

const rectangle = rectangleZonePoints({ x: 0.5, y: 0.5, width: 0.2, height: 0.1, rotation_deg: 0 });
assert.deepEqual(rectangle[0], { x: 0.4, y: 0.45 });
assert.deepEqual(rectangle[2], { x: 0.6, y: 0.55 });
assert.deepEqual(zoneCentroid(rectangle), { x: 0.5, y: 0.5 });
const shifted = translateZonePoints(rectangle, 0.7, 0);
assert.equal(Math.max(...shifted.map((point) => point.x)), 1);
assert.equal(removeZonePoint(rectangle, 0).length, 3);
assert.equal(removeZonePoint(rectangle.slice(0, 3), 0).length, 3);
assert.equal(moveZonePoint(rectangle, 1, { x: 2, y: -1 })[1].x, 1);
assert.equal(moveZonePoint(rectangle, 1, { x: 2, y: -1 })[1].y, 0);

const stacked = stackPropertyOverlays(
  [{ camera_id: "gate" }, { camera_id: "door" }],
  [{ key: "drive" }, { key: "yard" }],
  { kind: "camera", id: "gate" },
);
assert.deepEqual(stacked.map((entry) => entry.id), ["door", "drive", "yard", "gate"]);
const zoneOnTop = stackPropertyOverlays(
  [{ camera_id: "gate" }],
  [{ key: "drive" }, { key: "yard" }],
  { kind: "zone", id: "drive" },
);
assert.equal(zoneOnTop.at(-1).id, "drive");

const history = recordZonePointInsertion(recordZonePointInsertion([], 0), 1);
assert.deepEqual(history, [0, 1]);
const undone = undoZonePointInsertion([{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 0.2, y: 0.2 }], [0, 2]);
assert.equal(undone.points.length, 2);
assert.deepEqual(undone.history, [0]);
assert.equal(propertyLocationFields({
  property: { latitude: 1, longitude: 2, image: { latitude: 35.451111, longitude: -82.560556, address: "Home", width_m: 120 } },
  weather: { latitude: 9, longitude: 9 },
}).source, "image");
assert.equal(propertyLocationFields({
  property: { latitude: 35.1, longitude: -82.5, address: "Saved" },
  weather: { latitude: 9, longitude: 9 },
}).address, "Saved");

console.log("property view tests passed");
