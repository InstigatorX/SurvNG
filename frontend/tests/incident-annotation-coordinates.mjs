import assert from "node:assert/strict";
import test from "node:test";
import {
  incidentDetectionFrameSize,
  incidentImageRenderRect,
  incidentObjectFocusCropRect,
  incidentTrackingFrameSize,
} from "../src/incidentNavigation.mjs";

const hidden = {
  label: "person", track_id: 1, snapshot_visible: false,
  detection_frame_width: 896, detection_frame_height: 512,
  box: { x1: 100, y1: 100, x2: 120, y2: 150 },
};
const cover = {
  label: "person", track_id: 2, snapshot_visible: true,
  detection_frame_width: 3840, detection_frame_height: 2160,
  box: { x1: 2100, y1: 1110, x2: 2180, y2: 1300 },
};
const tracking = { frame_width: 896, frame_height: 512 };

function event(objects) {
  return { objects, object_tracking: tracking };
}

// SnapshotImage consumes these dimensions for both full-frame projection and
// server-crop reconstruction. A hidden object's dimensions must not leak in.
test("a hidden substream object cannot set the main cover coordinate plane", () => {
  const value = event([hidden, cover]);
  assert.deepEqual(incidentDetectionFrameSize(value), { width: 3840, height: 2160 });
  assert.deepEqual(incidentDetectionFrameSize(event([cover, hidden])), incidentDetectionFrameSize(value));
  assert.deepEqual(value.objects, [hidden, cover]); // No inventory mutation/reordering.
  assert.equal(hidden.snapshot_visible, false);
});

test("hidden main-stream history cannot set a visible substream coordinate plane", () => {
  const value = event([{ ...cover, snapshot_visible: false }, { ...hidden, snapshot_visible: true }]);
  assert.deepEqual(incidentDetectionFrameSize(value), { width: 896, height: 512 });
});

test("in-place cover replacement changes dimensions without moving the subject", () => {
  const value = event([hidden, { ...hidden, track_id: 2, snapshot_visible: true }]);
  assert.deepEqual(incidentDetectionFrameSize(value), { width: 896, height: 512 });
  Object.assign(value.objects[1], cover);
  assert.deepEqual(incidentDetectionFrameSize(value), { width: 3840, height: 2160 });
  assert.deepEqual(value.objects.map((object) => object.track_id), [1, 2]);
  assert.deepEqual(incidentTrackingFrameSize(value), { width: 896, height: 512 });
});

test("legacy objects without snapshot_visible retain their coordinate metadata", () => {
  assert.deepEqual(incidentDetectionFrameSize({ objects: [
    { detection_frame_width: "2560", detection_frame_height: "1920" },
  ] }), { width: 2560, height: 1920 });
  assert.deepEqual(incidentDetectionFrameSize(event([{ label: "person" }])), trackingSize());
  assert.equal(incidentDetectionFrameSize({ objects: [] }), null);
  assert.equal(incidentDetectionFrameSize(null), null);
});

function trackingSize() { return { width: tracking.frame_width, height: tracking.frame_height }; }

test("hidden dimensions are never a fallback when visible metadata is absent", () => {
  assert.equal(incidentDetectionFrameSize({ objects: [hidden] }), null);
  assert.deepEqual(incidentDetectionFrameSize(event([hidden])), trackingSize());
});

test("nonfinite and nonpositive dimensions do not poison visible annotation geometry", () => {
  for (const bad of [Infinity, "Infinity", NaN, "bad", 0, -1, null]) {
    for (const field of ["detection_frame_width", "detection_frame_height"]) {
      const invalid = { ...cover, [field]: bad };
      assert.deepEqual(incidentDetectionFrameSize(event([invalid, cover])), { width: 3840, height: 2160 });
    }
  }
});

test("the same cover box stays inside thumbnail and full-resolution render rectangles", () => {
  const size = incidentDetectionFrameSize(event([hidden, cover]));
  for (const raster of [{ width: 720, height: 405 }, { width: 3840, height: 2160 }]) {
    const rect = incidentImageRenderRect({ width: 384, height: 216 }, raster);
    const left = rect.x + cover.box.x1 * rect.width / size.width;
    const top = rect.y + cover.box.y1 * rect.height / size.height;
    const width = (cover.box.x2 - cover.box.x1) * rect.width / size.width;
    const height = (cover.box.y2 - cover.box.y1) * rect.height / size.height;
    assert.deepEqual({ left, top, width, height }, { left: 210, top: 111, width: 8, height: 19 });
    assert.ok(left + width <= rect.x + rect.width && top + height <= rect.y + rect.height);
  }
});

test("object-focus crop uses the visible cover plane, regardless of inventory order", () => {
  const expected = incidentObjectFocusCropRect(3840, 2160, [cover.box], 1, 16, 9);
  assert.ok(expected);
  for (const objects of [[hidden, cover], [cover, hidden]]) {
    const size = incidentDetectionFrameSize(event(objects));
    const crop = incidentObjectFocusCropRect(size.width, size.height, [cover.box], 1, 16, 9);
    assert.deepEqual(crop, expected);
    assert.ok(crop.x1 <= cover.box.x1 && crop.x2 >= cover.box.x2);
    assert.ok(crop.y1 <= cover.box.y1 && crop.y2 >= cover.box.y2);
  }
});

test("tracking replay keeps its independent substream dimensions after cover promotion", () => {
  assert.deepEqual(incidentTrackingFrameSize(event([hidden, cover])), trackingSize());
  assert.deepEqual(incidentTrackingFrameSize({ objects: [hidden, cover] }), { width: 3840, height: 2160 });
});
