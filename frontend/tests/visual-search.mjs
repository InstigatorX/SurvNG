import assert from "node:assert/strict";
import {
  appearanceCapableLabel,
  appearanceMatchesPath,
  hybridFindSimilarSubtitle,
  hybridMatchLabel,
  mergeHybridFindSimilarResults,
  resolveObjectTrackId,
  visualMatchLabel,
  visualSearchRequest,
} from "../src/visualSearch.mjs";

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

console.log("visual search helper tests passed");
