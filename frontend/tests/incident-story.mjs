import assert from "node:assert/strict";
import {
  cardIdentities, cardOrder, coverageChips, cropBackgroundStyle, estimatedBox, establishmentReasonText, followSegments, formatSpan, frameBoxes, incidentStoryTitle,
  layoutStoryLane, nearestObservation, objectKind, objectPath, objectSummary, reviewKeyAction, reviewPosition, storyObjects, storyline,
} from "../src/incidentStory.mjs";
import { BASE, CAMERAS, queueFor, storyIncident } from "./fixtures/incident-story.mjs";

const names = new Map(CAMERAS.map((camera) => [camera.id, camera.name]));
const cameraName = (id) => names.get(id) || id;
const incident = storyIncident();
const objects = storyObjects(incident);
const byLabel = Object.fromEntries(objects.map((object) => [object.label, object]));

// Kinds and inventory: uncertainty wins over the label; faces are not cards.
assert.equal(objectKind({ label: "person", certainty: "observed" }), "person");
assert.equal(objectKind({ label: "truck", certainty: "observed" }), "vehicle");
assert.equal(objectKind({ label: "dog", certainty: "possible" }), "uncertain");
assert.equal(objectKind({ label: "cat" }), "other");
assert.equal(storyObjects({ scene_objects: [{ id: "f", label: "face", observations: [] }] }).length, 0);
assert.deepEqual(cardOrder(objects).map((object) => object.label), ["person", "car", "dog"]);

// Title, summaries and spans.
assert.equal(incidentStoryTitle(incident, { cameraName, objects }), "Car arrives, person walks to Front Door");
assert.equal(incidentStoryTitle(incident, { cameraName, objects, people: ["Alex"] }), "Car arrives, Alex walks to Front Door");
assert.equal(incidentStoryTitle({ camera_id: "front-door", events: [] }, { cameraName }), "Motion at Front Door");
assert.equal(incidentStoryTitle({ camera_id: "front-door", events: [] }, { cameraName, fallback: "Alex detected at Front Door." }), "Alex detected at Front Door.");
assert.equal(incidentStoryTitle({ ...incident, scene_objects: [incident.scene_objects[2]] }, { cameraName }), "Possible dog in Lawn");
const drivewayPerson = { ...byLabel.person, intervals: byLabel.person.intervals.slice(0, 1) };
assert.equal(incidentStoryTitle(incident, { cameraName, objects: [drivewayPerson, { ...drivewayPerson, id: "second-track" }] }), "Person activity in Front Driveway");
assert.equal(formatSpan(102), "1 min 42 s");
assert.equal(formatSpan(12), "12 s");
const timeZone = "America/New_York";
assert.equal(objectSummary(byLabel.person, { cameraName, timeZone, identities: [] }), "Front Driveway · 10:24:41 AM");
assert.equal(objectSummary(byLabel.car, { cameraName, timeZone }), "Front Driveway · 10:24:20 AM");
const twoPeople = [{ kind: "person" }, { kind: "person" }, { kind: "vehicle" }];
assert.deepEqual(cardIdentities([{ kind: "person" }], [{ name: "Alex" }]), [{ name: "Alex" }]);
assert.equal(cardIdentities(twoPeople, [{ name: "Alex" }]), null, "one name is not put on two people");
assert.deepEqual(cardIdentities(twoPeople, []), [], "nobody recognized holds for every person");
assert.equal(objectSummary(byLabel.person, { cameraName, timeZone, identities: [{ name: "Alex" }] }), "Alex · Front Driveway · 10:24:41 AM");
assert.equal(objectSummary(byLabel.dog, { cameraName, timeZone }), "Side Yard · 10:25:29 AM");

// Storyline: one lane per episode, stationary continuation stays on the car's row.
const line = storyline(incident, objects);
assert.equal(line.start, BASE);
assert.equal(line.end, BASE + 102);
assert.deepEqual(line.lanes.map((lane) => lane.cameraId), ["front-driveway", "front-door", "side-yard"]);
const driveway = layoutStoryLane(line.lanes[0].bars, line.start, line.end, 500);
assert.equal(driveway.rows, 2);
assert.deepEqual(driveway.bars.filter(({ bar }) => bar.objectId === "object-car").map(({ bar, row }) => [bar.state, row]), [["moving", 0], ["stationary", 0]]);
assert.equal(driveway.bars.find(({ bar }) => bar.objectId === "object-person").row, 1);
const shortSightings = layoutStoryLane([
  { start: 10, end: 10 }, { start: 11, end: 11.2 }, { start: 98, end: 100 }, { start: 100, end: 100 },
], 0, 100, 220);
assert.ok(shortSightings.bars.every(({ width, left }) => width >= 24 && left + width <= 220));
assert.deepEqual(shortSightings.bars.map(({ row }) => row), [0, 1, 0, 1]);
assert.equal(line.lanes[2].bars[0].uncertain, true);
// Fallback without server intervals: one observed span per camera, never "stationary".
const legacy = storyObjects({ scene_objects: [{ ...incident.scene_objects[0], visible_intervals: undefined }] })[0];
assert.deepEqual(legacy.intervals.map((interval) => [interval.camera_id, interval.state]), [["front-driveway", "observed"], ["front-door", "observed"]]);
// A second episode on one camera is its own lane.
const repeated = storyline({ episodes: [{ id: "a", camera_id: "gate", start_at: "2026-01-01T00:00:00Z", end_at: "2026-01-01T00:01:00Z" }, { id: "b", camera_id: "gate", start_at: "2026-01-01T00:10:00Z", end_at: "2026-01-01T00:11:00Z" }] }, []);
assert.deepEqual(repeated.lanes.map((lane) => [lane.key, lane.repeated]), [["a", true], ["b", true]]);

// Path and follow plan.
const path = objectPath(byLabel.person, cameraName);
assert.deepEqual(path.map((step) => [step.number, step.camera, step.zones]), [[1, "Front Driveway", ["Driveway"]], [2, "Front Door", ["Entry"]]]);
assert.equal(path[0].note, "2 s overlap · associated automatically");
assert.equal(path[1].note, "Left view. Not seen on another camera.");
assert.match(objectPath({ ...byLabel.person, association_corrected: true }, cameraName)[0].note, /by a correction/);
assert.deepEqual(followSegments(byLabel.person, { start: BASE, end: BASE + 102 }).map((segment) => [segment.cameraId, segment.start - BASE, segment.end - BASE]),
  [["front-driveway", 19, 51], ["front-door", 45, 102]]);

// Frames: the nearest sighting on a camera, and every box in that capture.
assert.equal(nearestObservation(byLabel.person, BASE + 60, "front-door").id, "obs-person-door-62");
const cover = byLabel.car.observations.find((item) => item.id === "obs-cover-car");
assert.deepEqual(frameBoxes(objects, cover).map((item) => item.object.label).sort(), ["car", "person"]);
const otherFrames = [{
  observations: [
    { ...cover, captured_at: new Date((BASE + 30.533) * 1000).toISOString() },
    { ...cover, image: { source: "live" } },
  ],
}];
assert.equal(frameBoxes(otherFrames, cover).length, 0);
const crop = cropBackgroundStyle({ detection_frame_width: 1000, detection_frame_height: 500, box: { x1: 450, y1: 200, x2: 550, y2: 300 } }, 1, 0);
assert.deepEqual(crop, { backgroundSize: "1000% 500%", backgroundPosition: "50% 50%" });
assert.equal(cropBackgroundStyle({ box: { x1: 0, y1: 0, x2: 1, y2: 1 } }), null);

// Why: establishment reason names the zone and the cameras that joined.
assert.equal(establishmentReasonText(incident, cameraName), "Physical activity confirmed in the Driveway zone on Front Driveway. Front Door, Side Yard joined because their activity overlapped in time.");
assert.equal(establishmentReasonText({ ...incident, establishment: { status: "not_established" } }, cameraName), null);

// Review queue position and shortcuts.
const queue = queueFor("incident-story-1");
assert.deepEqual(reviewPosition(queue, "incident-story-1"), { index: 4, total: 25, decided: 4, remaining: 21, previous: "incident-q-3", next: "incident-q-5", nextUndecided: "incident-q-5" });
assert.equal(reviewPosition(null, "x").next, null);
const key = (keyName, target = { tagName: "BODY" }, extra = {}) => reviewKeyAction({ key: keyName, target, ...extra });
assert.equal(key("e"), "reviewed");
assert.equal(key("F"), "keep");
assert.equal(key("x"), "nothing_here");
assert.equal(key("j"), "next");
assert.equal(key("k"), "previous");
assert.equal(key("e", { tagName: "INPUT" }), null);
assert.equal(key("e", { tagName: "DIV", isContentEditable: true }), null);
assert.equal(key("e", { tagName: "BODY" }, { ctrlKey: true }), null);
assert.equal(key("q"), null);

// A followed object keeps a box on recording frames between analysed frames.
const doorEstimate = estimatedBox(byLabel.person, "front-door", BASE + 82);
assert.equal(doorEstimate.estimated, true);
assert.deepEqual(doorEstimate.box, { x1: 526, y1: 147, x2: 751, y2: 661 }, "halfway between the 62 s and 102 s boxes");
assert.equal(estimatedBox(byLabel.person, "front-door", BASE + 47.5).box.x1 > 470, true);
assert.equal(estimatedBox(byLabel.person, "side-yard", BASE + 82), null, "never borrowed from another camera");
assert.equal(estimatedBox({ ...byLabel.person, intervals: [] }, "front-door", BASE + 82), null, "a 40 s gap outside one sighting is not bridged");
assert.equal(estimatedBox(byLabel.dog, "side-yard", BASE + 100), null, "a lone observation far away is not reused");
assert.equal(estimatedBox(byLabel.person, "front-door", Number.NaN), null);

// Unfinished camera views are named; finished ones are not.
assert.deepEqual(coverageChips(incident, cameraName), [{ tone: "warn", text: "Side Yard analysis pending" }]);
assert.deepEqual(coverageChips({ episodes: [{ camera_id: "side-yard", coverage: { state: "incomplete", gaps: [{}] } }] }, cameraName), [{ tone: "warn", text: "Side Yard analysis incomplete" }]);
assert.deepEqual(coverageChips({ episodes: [{ camera_id: "front-door", coverage: { state: "sampled" } }] }, cameraName), []);
// Nothing is pending where recorded analysis is not coming: motion alone, or a view that was only sampled.
const waiting = { camera_id: "side-yard", coverage: { state: "incomplete", reason: "analysis pending", gaps: [] } };
assert.deepEqual(coverageChips({ episodes: [waiting] }, cameraName), []);
assert.deepEqual(coverageChips({ labels: ["person"], episodes: [waiting] }, cameraName), [{ tone: "warn", text: "Side Yard analysis pending" }]);
// Motion with recorded analysis queued is waiting on it like anything else.
assert.deepEqual(coverageChips({ episodes: [{ camera_id: "side-yard", coverage: { state: "incomplete", reason: "recorded analysis queued", gaps: [] } }] }, cameraName),
  [{ tone: "warn", text: "Side Yard analysis pending" }]);
for (const ended of ["detection turned off before recorded analysis finished", "failed", "analysis_window_limit"]) {
  assert.deepEqual(coverageChips({ episodes: [{ camera_id: "side-yard", coverage: { state: "incomplete", reason: ended, gaps: [] } }] }, cameraName), [], ended);
}
assert.deepEqual(coverageChips({ labels: ["person"], episodes: [{ camera_id: "side-yard", coverage: { state: "incomplete", gaps: [],
  reason: "sampled detector frames; continuous coverage not established" } }] }, cameraName), []);

console.log("incident story helpers passed");
