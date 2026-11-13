import assert from "node:assert/strict";
import { trackingCoverageLabel, containedFrameTransform, hlsPlaybackOffset, hlsProgramStartEpoch, incidentReplayTracking, incidentTrackingSource, playbackEpochAt, incidentTrackFramesAt, snapshotObjectTracks, snapshotTrackOverview, storedObjectTracks, trackFrameAt, trackPathD, withoutIsolatedTrackSpikes } from "../src/objectTrackReplay.mjs";
import { eventCoverEpoch } from "../src/incidentNavigation.mjs";

const tracks = storedObjectTracks({ object_tracking: { tracks: [{
  track_id: 7,
  label: "person",
  box: { x1: 20, y1: 10, x2: 50, y2: 80 },
  trajectory: [[11, 35, 45], [10, 25, 45], ["bad", 0, 0]],
  box_history: [[11, 20, 10, 50, 80], [10, 10, 10, 40, 80], [12, 5, 5, 4, 8]],
  reid_recovery_history: [
    { captured_at: 11, similarity: 0.91, resumed_completed_track: true, box: [20, 10, 50, 80] },
    { captured_at: "bad", similarity: 1, box: [0, 0, 1, 1] },
  ],
}] } });

assert.equal(tracks.length, 1);
assert.deepEqual(tracks[0].trajectory, [[10, 25, 45], [11, 35, 45]]);
assert.deepEqual(tracks[0].boxHistory, [[10, 10, 10, 40, 80], [11, 20, 10, 50, 80]]);
assert.deepEqual(tracks[0].recoveryHistory, [{ capturedAt: 11, similarity: 0.91, resumedCompletedTrack: true, box: [20, 10, 50, 80] }]);
assert.equal(trackFrameAt(tracks[0], 9.9), null);
assert.deepEqual(trackFrameAt(tracks[0], 10.5)?.box, [15, 10, 45, 80]);
assert.deepEqual(trackFrameAt(tracks[0], 10.5)?.path, [[25, 45], [30, 45]]);
assert.equal(trackFrameAt(tracks[0], 10.5)?.estimated, false);
assert.deepEqual(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.box, [20, 10, 50, 80]);
assert.equal(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.estimated, true);
assert.equal(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.recovery?.similarity, 0.91);
assert.equal(trackFrameAt(tracks[0], 12.1, { holdSeconds: 1, sampleFps: 2 }), null);

const sparseTrack = storedObjectTracks({ object_tracking: { tracks: [{
  track_id: 8,
  label: "car",
  box: { x1: 40, y1: 10, x2: 70, y2: 80 },
  trajectory: [[10, 25, 45], [14, 55, 45]],
  box_history: [[10, 10, 10, 40, 80], [14, 40, 10, 70, 80]],
}] } })[0];
assert.equal(trackFrameAt(sparseTrack, 10.5, { holdSeconds: 3, sampleFps: 2 })?.estimated, false);
assert.equal(trackFrameAt(sparseTrack, 12, { holdSeconds: 3, sampleFps: 2 })?.estimated, true);
assert.deepEqual(trackFrameAt(sparseTrack, 12, { holdSeconds: 3, sampleFps: 2 })?.box, [25, 10, 55, 80]);

const snapshotEvent = {
  created_at: "1970-01-01T00:00:11Z",
  snapshot_captured_at: "1970-01-01T00:00:10.500Z",
  object_tracking: { sample_fps: 2, tracks: [
    { ...tracks[0], track_id: 7, box_history: tracks[0].boxHistory },
    { ...tracks[0], track_id: 9, box_history: [[11, 20, 10, 50, 80]] },
    { ...tracks[0], track_id: 10, box_history: [[9, 20, 10, 50, 80]] },
    { ...tracks[0], track_id: 11, box_history: [] },
  ] },
};
const snapshotTracks = snapshotObjectTracks(snapshotEvent, eventCoverEpoch(snapshotEvent));
assert.deepEqual(snapshotTracks.map((track) => track.trackId), [7], "hide future, ended, and untimed tracks");
assert.deepEqual(snapshotTracks.map(({ x1, y1, x2, y2 }) => [x1, y1, x2, y2]), [[15, 10, 45, 80]], "use capture time rather than the trigger or final box");
assert.deepEqual(snapshotTracks[0].path, [[25, 45], [30, 45]], "exclude the future trajectory");
assert.deepEqual(snapshotObjectTracks(snapshotEvent, null), [], "unknown capture time cannot show tracks");
assert.deepEqual(snapshotObjectTracks({ object_tracking: { tracks: [{ ...sparseTrack, track_id: 8, box_history: sparseTrack.boxHistory }] } }, 12), [], "hide unobserved gaps");
assert.deepEqual(snapshotObjectTracks(snapshotEvent, 11).map((track) => track.trackId), [7, 9], "include tracks observed exactly at capture time");
const overview = snapshotTrackOverview(snapshotEvent, eventCoverEpoch(snapshotEvent));
assert.deepEqual(overview.map((track) => track.trackId), [7, 9, 10], "overview retains past and future tracks with timed observations");
assert.deepEqual(overview[0].paths, [[[25, 45], [35, 45]]], "overview shows the complete incident path beyond the snapshot time");
assert.deepEqual(overview.map((track) => track.snapshotBox), [[15, 10, 45, 80], null, null], "only current observations receive snapshot boxes");
assert.ok(snapshotTrackOverview(snapshotEvent, null).every((track) => track.snapshotBox === null), "unknown capture time still permits paths, never stale boxes");

assert.deepEqual(storedObjectTracks({ object_tracking: { tracks: [{ track_id: 1, label: "person", box: {} }] } }), []);

const earlierTracked = {
  id: 10,
  created_at: "2026-07-28T22:36:29+00:00",
  object_tracking: { tracks: [{ track_id: 1 }] },
};
const laterTracked = {
  id: 12,
  created_at: "2026-07-28T22:36:49+00:00",
  object_tracking: { tracks: [{ track_id: 2 }] },
};
const selectedChild = {
  id: 11,
  created_at: "2026-07-28T22:36:31+00:00",
  start_epoch: Date.parse("2026-07-28T22:30:00+00:00") / 1000,
  events: [laterTracked, earlierTracked],
};
assert.equal(incidentTrackingSource(selectedChild)?.id, 10);
assert.equal(incidentTrackingSource({
  ...selectedChild,
  created_at: "2026-07-28T22:36:47+00:00",
})?.id, 12);
assert.equal(incidentTrackingSource(earlierTracked), earlierTracked);
assert.equal(incidentTrackingSource({ id: 13, events: [] }), null);
assert.equal(
  incidentTrackingSource({ id: 14, events: [] }, { events: [earlierTracked] })?.id,
  10,
);

assert.equal(playbackEpochAt(1000, 8, 8), 1000);
assert.equal(playbackEpochAt(1000, 29, 8), 1021);
assert.equal(playbackEpochAt(1000, 0, 0), 1000);
assert.equal(playbackEpochAt(1000, "bad", 0), null);

const eventManifest = `#EXTM3U
#EXT-X-START:TIME-OFFSET=7.000,PRECISE=YES
#EXT-X-PROGRAM-DATE-TIME:2026-07-31T11:57:13+00:00
#EXTINF:10.000,
segment.m4s
`;
assert.equal(hlsProgramStartEpoch(eventManifest), Date.parse("2026-07-31T11:57:13+00:00") / 1000);
assert.equal(hlsProgramStartEpoch("#EXTM3U\n#EXTINF:10,"), null);
assert.equal(hlsProgramStartEpoch(null), null);
assert.equal(hlsPlaybackOffset(1020, 1013, 19), 26);
assert.equal(hlsPlaybackOffset(1020, 1020, 0), 0);
assert.equal(hlsPlaybackOffset(1020, null, 4), 4);

const cleanTrackSamples = withoutIsolatedTrackSpikes([
  [1.0, 570, 532, 1074, 1858],
  [1.31, 1120, 312, 2044, 1916],
  [1.63, 600, 500, 1204, 1916],
]);
assert.deepEqual(cleanTrackSamples.map((sample) => sample[0]), [1.0, 1.63]);
assert.deepEqual(withoutIsolatedTrackSpikes([
  [1.0, 0, 0, 100, 200],
  [1.33, 25, 0, 125, 200],
  [1.66, 50, 0, 150, 200],
]), [
  [1.0, 0, 0, 100, 200],
  [1.33, 25, 0, 125, 200],
  [1.66, 50, 0, 150, 200],
]);

assert.deepEqual(containedFrameTransform({ width: 1600, height: 900 }, { width: 1920, height: 1080 }), {
  x: 0,
  y: 0,
  width: 1600,
  height: 900,
  scale: 5 / 6,
});
assert.deepEqual(containedFrameTransform({ width: 1600, height: 900 }, { width: 1920, height: 2560 }), {
  x: 462.5,
  y: 0,
  width: 675,
  height: 900,
  scale: 0.3515625,
});
assert.equal(containedFrameTransform({ width: 0, height: 900 }, { width: 1920, height: 1080 }), null);

console.log("object track replay tests passed");

assert.equal(trackingCoverageLabel({ state: "active" }), "Tracking processing");
assert.equal(trackingCoverageLabel({ state: "complete", coverage_incomplete: false }), "Tracking complete");
assert.match(trackingCoverageLabel({ state: "interrupted", completion_reason: "missing_media_while_object_active" }), /recording coverage unavailable/);
assert.match(trackingCoverageLabel({ state: "interrupted", completion_reason: "processing_budget_exhausted" }), /processing time limit/);
assert.equal(trackingCoverageLabel({ state: "complete", coverage_incomplete: true }), "Tracking incomplete");

const { recordedIncidentWindow, trackingCoverageAt } = await import("../src/objectTrackReplay.mjs");
const windowTracking = {
  window_start_epoch: 100, window_end_epoch: 130,
  analyzed_from: new Date(100000).toISOString(), analyzed_through: new Date(110000).toISOString(), state: "active",
};
assert.deepEqual(recordedIncidentWindow({created_at: new Date(110000).toISOString(), object_tracking: windowTracking}, 5, 5), {start:100, end:130});
assert.equal(trackingCoverageAt(windowTracking, 105), null);
assert.equal(trackingCoverageAt(windowTracking, 115), "Tracking pending for this footage");
assert.equal(trackingCoverageAt({...windowTracking, state: "interrupted"}, 115), "This footage has not been analyzed");
assert.equal(trackingCoverageAt({}, 115), null); // Legacy stored sessions retain their existing presentation.
assert.equal(trackFrameAt({boxHistory:[[100,0,0,10,10]], trajectory:[]}, 101, {holdSeconds:5, observationsOnly:true}), null);
assert.equal(incidentTrackingSource({object_tracking: {state:"active", tracks:[]}})?.object_tracking.state, "active");
const replay = incidentReplayTracking(
  { id: 8, object_tracking: { state: "complete", tracks: [], frame_width: 640, frame_height: 360, sample_fps: 3 } },
  { scene_objects: [{ label: "person", observations: [
    { scene_track_key: "tracking:8:run:1", label: "person", captured_at_epoch: 10, detection_frame_width: 1280, detection_frame_height: 720, box: { x1: 1, y1: 2, x2: 30, y2: 40 } },
    { scene_track_key: "tracking:8:run:1", label: "person", captured_at_epoch: 11, detection_frame_width: 1280, detection_frame_height: 720, box: { x1: 4, y1: 5, x2: 34, y2: 44 } },
    { scene_track_key: "tracking:8:run:1", label: "person", captured_at_epoch: 10.2, detection_frame_width: 640, detection_frame_height: 360, box: { x1: 1, y1: 1, x2: 8, y2: 8 } },
    { label: "person", captured_at_epoch: 12, detection_frame_width: 1280, detection_frame_height: 720, box: { x1: 9, y1: 9, x2: 20, y2: 20 } },
  ] }] },
);
assert.equal(storedObjectTracks(replay).length, 1);
assert.deepEqual(replay.object_tracking.tracks[0].box_history, [[10, 1, 2, 30, 40], [10.2, 2, 2, 16, 16], [11, 4, 5, 34, 44]]);
assert.equal(replay.object_tracking.frame_width, 1280);
assert.equal(replay.object_tracking.state, "complete");
const stored = { id: 9, object_tracking: { state: "complete", tracks: [{ track_id: 4, label: "car", box: { x1: 1, y1: 1, x2: 5, y2: 6 }, box_history: [[1, 1, 1, 5, 6]] }] } };
const laterScene = incidentReplayTracking({ ...stored, object_tracking: { ...stored.object_tracking, frame_width: 100, frame_height: 100 } }, { scene_objects: [{ observations: [{ scene_track_key: "9:tracking:9:run:1", label: "person", captured_at_epoch: 2, detection_frame_width: 100, detection_frame_height: 100, box: { x1: 0, y1: 0, x2: 2, y2: 2 } }] }] });
assert.equal(laterScene.object_tracking.tracks.length, 2, "later scene observations must not hide the earlier stored fragment");
const partial = {
  id: 10,
  object_tracking: {
    state: "complete",
    window_start_epoch: 100,
    window_end_epoch: 160,
    analyzed_from: new Date(150 * 1000).toISOString(),
    analyzed_through: new Date(160 * 1000).toISOString(),
    tracks: [{ track_id: 1, label: "person", box: { x1: 4, y1: 5, x2: 34, y2: 44 }, box_history: [[150, 4, 5, 34, 44]] }],
  },
};
const repaired = incidentReplayTracking(partial, { scene_objects: [{ label: "person", observations: [
  { scene_track_key: "tracking:10:run:1", label: "person", captured_at_epoch: 110, detection_frame_width: 100, detection_frame_height: 100, box: { x1: 1, y1: 2, x2: 20, y2: 40 } },
  { scene_track_key: "tracking:10:run:1", label: "person", captured_at_epoch: 150, detection_frame_width: 100, detection_frame_height: 100, box: { x1: 4, y1: 5, x2: 34, y2: 44 } },
] }] });
assert.equal(repaired.object_tracking.tracks[0].box_history[0][0], 110);
assert.equal(Date.parse(repaired.object_tracking.analyzed_from) / 1000, 110);
assert.equal(trackingCoverageAt(repaired.object_tracking, 110), null);

// The modal uses a representative event/episode clip, whose saved tracking
// contains only the final fragment. Recover the cover's earlier observations
// from the parent scene without replacing the displayed snapshot metadata.
const earlyCover = { ...partial, created_at: new Date(150000).toISOString(), snapshot_captured_at: new Date(110000).toISOString() };
assert.deepEqual(snapshotObjectTracks(earlyCover, eventCoverEpoch(earlyCover)), []);
const coverReplay = incidentReplayTracking(earlyCover, { scene_objects: [{ label: "person", observations: [
  { scene_track_key: "10:tracking:10:run:1", label: "person", captured_at_epoch: 110, detection_frame_width: 100, detection_frame_height: 100, box: { x1: 1, y1: 2, x2: 20, y2: 40 } },
  { scene_track_key: "10:tracking:10:run:1", label: "person", captured_at_epoch: 150, detection_frame_width: 100, detection_frame_height: 100, box: { x1: 4, y1: 5, x2: 34, y2: 44 } },
] }] });
const recoveredCover = { ...earlyCover, object_tracking: coverReplay.object_tracking };
assert.equal(recoveredCover.snapshot_captured_at, earlyCover.snapshot_captured_at);
assert.deepEqual(snapshotObjectTracks(recoveredCover, eventCoverEpoch(recoveredCover)).map(({ x1, y1, x2, y2 }) => [x1, y1, x2, y2]), [[1, 2, 20, 40]]);

const fragment = (id, camera, epoch, x = 10) => ({
  id, camera_id: camera,
  object_tracking: { frame_width: 100, frame_height: 100, sample_fps: 2, state: "complete", tracks: [{
    track_id: 1, label: "person", box: { x1: x, y1: 10, x2: x + 10, y2: 30 },
    box_history: [[epoch, x, 10, x + 10, 30]],
  }] },
});
const firstFragment = fragment(20, "downstairs", 10);
const nextFragment = fragment(21, "downstairs", 20);
const otherCamera = fragment(22, "gate", 10);
const observation = (key, eventId, camera, epoch, size = 100) => ({
  scene_track_key: key, event_id: eventId, camera_id: camera, label: "person", captured_at_epoch: epoch,
  detection_frame_width: size, detection_frame_height: size,
  box: { x1: size / 10, y1: size / 10, x2: size / 5, y2: size * 0.3 },
});
const wholeIncident = {
  camera_id: "downstairs", events: [firstFragment, nextFragment, otherCamera], scene_objects: [{ observations: [
    observation("20:tracking:20:run-a:1", 20, "downstairs", 10.0004),
    observation("tracking:20:run-a:1", 20, "downstairs", 11, 200),
    observation("tracking:20:run-b:1", 20, "downstairs", 15),
    observation("tracking:22:run-a:1", 22, "gate", 10),
  ] }],
};
const assembled = incidentReplayTracking(firstFragment, wholeIncident);
const assembledTracks = assembled.object_tracking.tracks;
assert.equal(assembledTracks.length, 3, "merge an overlapping fragment, retain another event and separate reused IDs across runs");
assert.equal(new Set(assembledTracks.map((track) => track.track_id)).size, 3);
assert.ok(assembledTracks.every((track) => !track.replay_key.includes("22:")), "exclude other cameras");
const mergedTrack = assembledTracks.find((track) => track.replay_key === "tracking:20:run-a:1");
assert.equal(mergedTrack.box_history.length, 2, "deduplicate rounded timestamps and prefixed scene keys");
assert.deepEqual(mergedTrack.box_history[0].slice(1), mergedTrack.box_history[1].slice(1), "normalize observations at different resolutions");
const laterEpisode = incidentReplayTracking({ ...nextFragment, events: [nextFragment], scene_clip_window: { episode_id: "later" } }, wholeIncident);
assert.equal(laterEpisode.object_tracking.tracks.length, 1, "exclude other episodes on the same camera");
const refreshed = incidentReplayTracking(firstFragment, { ...wholeIncident, events: [...wholeIncident.events].reverse() });
assert.deepEqual(refreshed.object_tracking.tracks.map((track) => track.track_id), assembledTracks.map((track) => track.track_id), "identities survive response ordering changes");
const repeated = incidentReplayTracking(assembled, wholeIncident);
assert.equal(repeated.object_tracking.tracks.length, 3, "reassembling a replay does not duplicate its tracks");
assert.equal(Date.parse(assembled.object_tracking.analyzed_from) / 1000, 10);
assert.equal(Date.parse(assembled.object_tracking.analyzed_through) / 1000, 20);

const pathTrack = storedObjectTracks({ object_tracking: { tracks: [{
  track_id: 1, label: "person", box: { x1: 40, y1: 0, x2: 50, y2: 10 },
  box_history: [[10, 0, 0, 10, 10], [11, 10, 0, 20, 10], [14, 40, 0, 50, 10]],
}] } });
assert.deepEqual(incidentTrackFramesAt(pathTrack, 9), [], "hide future paths");
const endedPath = incidentTrackFramesAt(pathTrack, 16, { holdSeconds: 0, observationsOnly: true });
assert.equal(endedPath[0].box, null, "completed tracks retain no current box");
assert.deepEqual(endedPath[0].paths, [[[5, 5], [15, 5]], [[45, 5]]], "retain completed trails and break across missing observations");
assert.deepEqual(incidentTrackFramesAt(pathTrack, 10, { holdSeconds: 0 })[0].paths[0][0], [5, 5], "seeking backwards excludes later history");

assert.equal(trackPathD([]), "");
assert.equal(trackPathD([[0, 0], [0, 0]]), "M 0 0");
assert.equal(trackPathD([[0, 0], [10, 10]]), "M 0 0 L 10 10", "two observations remain a straight segment");
assert.equal(trackPathD([[0, 0], [10, 0], [10, 10]], 2), "M 0 0 L 8 0 Q 10 0 10 2 L 10 10", "round a turn while preserving endpoints");
assert.equal(trackPathD([[0, 0], [10, 0], [10, 0], [10, 10]], 2), "M 0 0 L 8 0 Q 10 0 10 2 L 10 10", "stationary samples do not produce invalid curves");
assert.equal(trackPathD([[0, 0], [3, 0], [3, 3]], 50), "M 0 0 L 2 0 Q 3 0 3 1 L 3 3", "limit rounding to local segment lengths");
