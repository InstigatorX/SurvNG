import assert from "node:assert/strict";
import { incidentEpisodeClips, incidentEpisodeMediaUrl, incidentSceneObjects, observationBoxStyle, observationImageDescription, observedObjectSummaries, sceneActivityText, sceneCoverageText, sceneEstablishmentText, sceneNotificationSummary, sceneObjectDetection, sceneObjectLabel, canonicalIncidentHref } from "../src/incidentScene.mjs";
import { incidentDetailQuery } from "../src/incidentNavigation.mjs";

const canonical = [{ id: "person-1", label: "person", certainty: "supported", observations: [] }];
assert.deepEqual(sceneObjectDetection({ label: "robot_lawnmower", confidence: 0.433 }), { label: "robot lawnmower", confidence: 0.433 });
assert.deepEqual(sceneObjectDetection({ label: "person", observations: [{ confidence: 0.2 }, { confidence: 0.75 }] }), { label: "person", confidence: 0.75 });
assert.deepEqual(observedObjectSummaries([
  { label: "person", observations: [{ confidence: 0.76 }, { confidence: 0.49 }] },
  { label: "person", confidence: 0.81 },
  { label: "car", observations: [{ confidence: 0.32 }, { confidence: 0.54 }, { confidence: 0.43 }, { confidence: 0.70 }] },
]), [
  { label: "person", confidence: 0.76 },
  { label: "car", confidence: 0.485 },
]);
assert.equal(sceneObjectLabel({ label: "person", certainty: "possible" }), "Possible person");
assert.equal(sceneObjectLabel({ label: "person", certainty: "supported", continuity_uncertain: true }), "person");
assert.equal(sceneEstablishmentText(undefined), null);
assert.match(sceneEstablishmentText({ status: "not_established" }).explanation, /preserves.*observations/);
assert.match(sceneEstablishmentText({ status: "incomplete" }).explanation, /does not establish that nothing happened/);
assert.match(sceneEstablishmentText({ status: "unverified" }).title, /unverified/);
assert.match(observationImageDescription({ width: 896, height: 512, source: "live_substream", analyzed_frame: true }), /896 × 512 · live substream · Analyzed frame/);
assert.match(observationImageDescription({ width: 2560, height: 1440, analyzed_frame: false }), /not the analyzed frame/);
assert.match(observationImageDescription(undefined), /provenance unknown/);
assert.equal(incidentSceneObjects({ scene_objects: canonical, objects: [{ label: "car" }] }), canonical);
assert.deepEqual(incidentSceneObjects({ scene_objects: [], objects: [{ label: "car" }] }), []);
const legacy = incidentSceneObjects({ events: [
  { id: 1, objects: [{ label: "person", incident_eligible: false }] },
  { id: 2, objects: [{}, { label: "car", snapshot_visible: false }] },
] });
assert.deepEqual(legacy.map((object) => object.label), ["person", "car"]);
assert.equal(legacy[1].object_index, 1);
assert.equal(legacy[1].observations[0].snapshot_available, false, "do not substitute a cover as supporting evidence");
assert.equal(legacy[0].certainty, "uncertain", "historical lack of confirmation stays explicit");
assert.deepEqual(observationBoxStyle({ detection_frame_width: 1280, detection_frame_height: 800, box: { x1: 128, y1: 160, x2: 640, y2: 560 } }), { left: "10%", top: "20%", width: "40%", height: "50%" });
assert.equal(observationBoxStyle({ box: { x1: 0, y1: 0, x2: 100, y2: 100 } }), null);
assert.match(sceneCoverageText({ state: "sampled", gaps: [{ reason: "missing_video" }] }), /incomplete/);
assert.match(sceneCoverageText({ state: "historical" }), /unknown/);
assert.equal(canonicalIncidentHref({ incident_id: "scene-1" }), "/incidents?incident_id=scene-1");
assert.equal(incidentDetailQuery({ incident_id: "scene-1", revision: 2, events: [] }), "incident_id=scene-1");
const start = "2026-09-24T20:25:00Z";
const clips = incidentEpisodeClips({ camera_id: "a", events: [
  { id: 1, camera_id: "a", created_at: start, object_tracking: { window_start_epoch: 0, window_end_epoch: 9999999999 } },
  { id: 2, camera_id: "b", created_at: "2026-09-24T20:26:00Z" },
], episodes: [
  { id: "second", camera_id: "b", start_at: "2026-09-24T20:26:00Z", end_at: "2026-09-24T20:26:12Z", event_ids: [2] },
  { id: "first", camera_id: "a", start_at: start, end_at: "2026-09-24T20:25:10Z", event_ids: [1] },
  { id: "missing", camera_id: "c", start_at: start, end_at: start, event_ids: [999] },
] });
assert.deepEqual(clips.filter((episode) => episode.clip).map((episode) => episode.clip.representative_event_id), [1, 2]);
assert.equal(clips[0].clip.scene_clip_window.end - clips[0].clip.scene_clip_window.start, 10);
assert.equal(clips.find((episode) => episode.id === "second").clip.events.length, 1, "other-camera events never expand an episode clip");
assert.equal(clips.find((episode) => episode.id === "missing").clip, null);
const longEpisode = incidentEpisodeClips({ id: 1, camera_id: "a", created_at: start, start_at: start, end_at: "2026-09-24T22:25:00Z" })[0];
assert.equal(longEpisode.clip, null, "never silently truncate a long episode at the server's one-hour bound");
assert.match(longEpisode.playback_note, /recording timeline/);
const longClips = incidentEpisodeClips({ camera_id: "a", events: [{ id: 1, camera_id: "a", created_at: start }], episodes: [{ id: "long", camera_id: "a", start_at: start, end_at: "2026-09-24T22:25:00Z", event_ids: [1] }] });
assert.equal(longClips.length, 8, "two hours play as bounded fifteen-minute requests");
for (let index = 0; index < longClips.length; index += 1) {
  const bounds = longClips[index].clip.scene_clip_window;
  assert.equal(bounds.episode_id, "long");
  assert.equal(bounds.end - bounds.start, 900);
  assert.equal(bounds.start, Date.parse(start) / 1000 + index * 900);
  assert.equal(longClips[index].clip.representative_event_id, 1);
}
const chunkUrl = new URL(incidentEpisodeMediaUrl("/api/events/1/clip.mp4?source=main", longClips[7].clip), "http://localhost");
assert.equal(chunkUrl.searchParams.get("episode_id"), "long");
assert.equal(Number(chunkUrl.searchParams.get("start_epoch")), Date.parse(start) / 1000 + 6300);
assert.equal(incidentEpisodeMediaUrl("/legacy.mp4", { scene_clip_window: { start: 0, end: 10 } }), "/legacy.mp4");
assert.equal(sceneActivityText({ kind: "last_seen", label: "person" }), "person last observed");
assert.equal(sceneActivityText({ kind: "zone_changed", label: "person", from_zones: ["Road"], to_zones: ["Driveway"] }), "person zone changed: Road → Driveway");
assert.match(sceneNotificationSummary([{ eligible: false, objects: [{ label: "person", eligible: false }] }]), /^No observations met/);
assert.match(sceneNotificationSummary([{ eligible: true }]), /^Some observations met/);
console.log("incident scene inventory, source geometry, coverage, and canonical navigation tests passed");
