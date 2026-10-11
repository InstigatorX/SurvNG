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
  overlayPresentation,
  pruneHighlights,
  rotationFromPointer,
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

console.log("property view tests passed");
