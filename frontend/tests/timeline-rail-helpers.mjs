import assert from "node:assert/strict";
import {
  condenseStory, crossCameraLinks, formatQuietDuration, layoutRailCards, nextStoryEvent, normalizedRailMinutes,
  normalizedTimelineView, railDensityBuckets, railEventKind, railEventTitle, railTickSeconds, storyEventAt,
} from "../src/timelineRail.mjs";

assert.equal(normalizedRailMinutes("15"), 15);
assert.equal(normalizedRailMinutes(45), 60, "unknown spans fall back to one hour");
assert.equal(normalizedTimelineView("cameras"), "cameras");
assert.equal(normalizedTimelineView("nope"), "rail");
assert.equal(normalizedTimelineView("classic"), "rail", "the retired classic view falls back to the rail");

assert.equal(railEventKind({ has_objects: true, labels: ["person"] }), "person");
assert.equal(railEventKind({ has_objects: true, labels: ["Truck"] }), "vehicle");
assert.equal(railEventKind({ has_objects: true, labels: ["dog"] }), "object");
assert.equal(railEventKind({ has_objects: false, labels: [] }), "motion");
assert.equal(railEventTitle({ has_objects: true, labels: ["person", "car", "dog"] }), "Person · Car +1");
assert.equal(railEventTitle({ has_objects: false }), "Motion");
assert.equal(railEventTitle({ has_objects: true, labels: ["person"], primary_identity: { id: 4, display_name: "Alex" } }), "Alex");

// Labelled ticks stay at least 44 px apart at any zoom.
assert.equal(railTickSeconds(0.5), 120);
assert.equal(railTickSeconds(0.05), 900);
assert.equal(railTickSeconds(0), 3600);

// Newest first; a card that would overlap the one above joins it.
const y = (epoch) => (1000 - epoch) * 2;
const cards = layoutRailCards([
  { id: 1, incident_epoch: 100 }, { id: 2, incident_epoch: 900 }, { id: 3, incident_epoch: 880 }, { id: 4, incident_epoch: 500 },
], y);
assert.deepEqual(cards.map((card) => card.events.map((event) => event.id)), [[2, 3], [4], [1]]);
assert.equal(cards[0].top, y(900) - 30);

const buckets = railDensityBuckets(0, 3600, [
  { incident_epoch: 100, has_objects: true }, { incident_epoch: 200, has_objects: false }, { incident_epoch: 1000, has_objects: false },
], [{ start_epoch: 0, end_epoch: 2000 }]);
assert.equal(buckets.length, 4);
assert.deepEqual(buckets.map((bucket) => bucket.kind), ["object", "motion", "recorded", "empty"]);
assert.equal(buckets[0].level, 1);

const story = condenseStory([
  { id: "a", incident_epoch: 1000, last_epoch: 1030 },
  { id: "b", incident_epoch: 1100, last_epoch: 1120 },
  { id: "c", incident_epoch: 5000, last_epoch: 5010 },
], { quietSeconds: 300, startEpoch: 0, endEpoch: 9000 });
assert.deepEqual(story.map((item) => item.type === "quiet" ? `quiet:${item.seconds}` : item.event.id), ["quiet:1000", "a", "b", "quiet:3880", "c", "quiet:3990"]);
assert.equal(storyEventAt(story, 1105)?.id, "b");
assert.equal(storyEventAt(story, 2000), null, "a quiet stretch has no current event");
const overlapping = condenseStory([{ id: "long", incident_epoch: 100, last_epoch: 400 }, { id: "short", incident_epoch: 200, last_epoch: 220 }], { quietSeconds: 300 });
assert.equal(storyEventAt(overlapping, 210)?.id, "long");
assert.equal(storyEventAt(overlapping, 210, 3, "short")?.id, "short", "the user's pick wins while it covers the playhead");
assert.equal(storyEventAt(overlapping, 300, 3, "short")?.id, "long", "a pick that no longer covers the playhead is not forced");
assert.equal(nextStoryEvent(story, 1125)?.id, "c", "auto-advance skips the quiet stretch");
assert.equal(nextStoryEvent(story, 6000), null);
assert.equal(formatQuietDuration(3880), "1 h 5 m");
assert.equal(formatQuietDuration(240), "4 m");

const links = crossCameraLinks([
  { id: 1, camera_id: "gate", incident_epoch: 100, primary_identity: { id: 7 } },
  { id: 2, camera_id: "front", incident_epoch: 150, primary_identity: { id: 7 } },
  { id: 3, camera_id: "front", incident_epoch: 160, primary_identity: { id: 8 } },
  { id: 4, camera_id: "gate", incident_epoch: 900, primary_identity: { id: 7 } },
]);
assert.deepEqual(links.map((link) => [link.from.id, link.to.id]), [[1, 2]], "only the same identity on another camera within two minutes links");

console.log("timeline rail helper tests passed");
