import assert from "node:assert/strict";
import {
  appearanceCapableLabel,
  appearanceMatchesPath,
  drawContainedCropPreview,
  hybridFindSimilarSubtitle,
  hybridMatchLabel,
  isValidObjectIndex,
  mergeHybridFindSimilarResults,
  normalizeVisualFrameCrop,
  resolveObjectTrackId,
  semanticResultThumbnailUrl,
  visualFrameCropFromPoints,
  visualFrameSearchRequest,
  visualMatchLabel,
  visualSearchRequest,
} from "../src/visualSearch.mjs";

assert.equal(isValidObjectIndex(null), false);
assert.equal(isValidObjectIndex(undefined), false);
assert.equal(isValidObjectIndex(""), false);
assert.equal(isValidObjectIndex(0), true);
assert.equal(isValidObjectIndex("0"), true);
assert.equal(isValidObjectIndex(1), true);
assert.equal(isValidObjectIndex(-1), false);
assert.equal(isValidObjectIndex(1.5), false);

assert.deepEqual(visualSearchRequest({
  eventId: 42,
  objectIndex: 1,
  cameraFilter: "gate",
  objectFilter: "person",
  startAt: "start",
  endAt: "end",
  sourceKinds: ["object_crop"],
}), {
  event_id: 42,
  object_index: 1,
  camera_ids: ["gate"],
  object_labels: ["person"],
  start_at: "start",
  end_at: "end",
  limit: 50,
  source_kinds: ["object_crop"],
  exclude_anchor: true,
});

assert.deepEqual(
  visualFrameCropFromPoints({ x: 0.8, y: 0.7 }, { x: 0.2, y: 0.1 }),
  { x: 0.2, y: 0.1, width: 0.6000000000000001, height: 0.6 },
);
assert.deepEqual(
  normalizeVisualFrameCrop({ x: 0.9, y: 0.8, width: 0.4, height: 0.5 }),
  { x: 0.9, y: 0.8, width: 0.09999999999999998, height: 0.19999999999999996 },
);
assert.equal(
  visualFrameCropFromPoints({ x: 0.1, y: 0.1 }, { x: 0.101, y: 0.102 }),
  null,
);
assert.deepEqual(visualFrameSearchRequest({
  cameraId: "gate",
  epoch: 123.456,
  source: "live",
  crop: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
  cameraFilter: "yard",
  sourceKinds: ["object_crop"],
  excludeEventId: 42,
}), {
  camera_id: "gate",
  epoch: 123.456,
  source: "live",
  x: 0.1,
  y: 0.2,
  width: 0.3,
  height: 0.4,
  camera_ids: ["yard"],
  object_labels: [],
  start_at: "",
  end_at: "",
  limit: 50,
  minimum_score: -1,
  source_kinds: ["object_crop"],
  exclude_event_id: 42,
});

assert.equal(visualMatchLabel("strong_match"), "Strong match");
assert.equal(visualMatchLabel("possible_match"), "Possible match");
assert.equal(visualMatchLabel(""), "Visually similar");

assert.equal(appearanceCapableLabel("person"), true);
assert.equal(appearanceCapableLabel("car"), true);
assert.equal(appearanceCapableLabel("dog"), false);
assert.equal(appearanceCapableLabel("package"), false);

assert.equal(resolveObjectTrackId({ track_id: 7 }), 7);
assert.equal(resolveObjectTrackId({ label: "person" }, {
  object_tracking: { tracks: [{ track_id: 3, label: "person" }] },
}), 3);
assert.equal(resolveObjectTrackId({ label: "person" }, {
  object_tracking: {
    cover_primary_track_id: 9,
    tracks: [
      { track_id: 8, label: "person" },
      { track_id: 9, label: "person" },
    ],
  },
}), 9);

assert.equal(
  appearanceMatchesPath(9, { hours: 12, limit: 8, trackId: 3, crossCameraOnly: false }),
  "/api/events/9/appearance-matches?hours=12&limit=8&cross_camera_only=false&track_id=3",
);

const merged = mergeHybridFindSimilarResults({
  appearanceMatches: [
    {
      event_id: 1,
      camera_id: "a",
      created_at: "t1",
      similarity: 0.91,
      visually_similar: true,
      candidate_label: "person",
      anchor_track_id: 2,
      candidate_track_id: 4,
    },
    {
      event_id: 2,
      camera_id: "b",
      created_at: "t2",
      similarity: 0.4,
      visually_similar: false,
    },
  ],
  visualResults: [
    {
      score: 0.8,
      match_strength: "visual_similarity",
      event: { id: 1, camera_id: "a", created_at: "t1" },
    },
    {
      score: 0.7,
      match_strength: "possible_match",
      event: { id: 3, camera_id: "c", created_at: "t3" },
    },
    {
      score: 0.69,
      match_strength: "possible_match",
      event: { id: 4, camera_id: "c", created_at: "t3" },
    },
  ],
  limit: 8,
});

assert.deepEqual(merged.map((item) => [item.query_mode, item.event.id]), [
  ["appearance", 1],
  ["visual", 3],
]);
assert.equal(hybridMatchLabel(merged[0]), "Appearance 91%");
assert.equal(hybridMatchLabel(merged[1]), "Possible match");
assert.match(
  hybridFindSimilarSubtitle({
    objectLabel: "person",
    usedAppearance: true,
    usedVisual: true,
    trackId: 2,
  }),
  /Appearance trail for person #2/,
);

assert.equal(
  semanticResultThumbnailUrl({ thumbnail_url: "/api/events/9/thumbnail.jpg?width=320" }),
  "/api/events/9/thumbnail.jpg?width=320",
);
assert.match(
  semanticResultThumbnailUrl({
    event: { id: 12 },
    evidence: { bbox: [10, 20, 110, 90] },
  }, 240, 80),
  /^\/api\/events\/12\/thumbnail\.jpg\?.*focus_bbox=10%2C20%2C110%2C90/,
);
assert.match(
  semanticResultThumbnailUrl({ event: { id: 7 } }),
  /^\/api\/events\/7\/thumbnail\.jpg\?.*object_focus=true/,
);

const previewState = { cleared: false, drawArgs: null };
const previewCanvas = {
  width: 176,
  height: 121,
  getContext() {
    return {
      clearRect() { previewState.cleared = true; },
      drawImage(...args) { previewState.drawArgs = args; },
    };
  },
};
const previewImage = { naturalWidth: 1920, naturalHeight: 1080 };
drawContainedCropPreview(previewCanvas, previewImage, { x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
assert.equal(previewState.cleared, true);
assert.deepEqual(previewState.drawArgs?.slice(0, 5), [previewImage, 480, 270, 960, 540]);

console.log("visual search helper tests passed");
