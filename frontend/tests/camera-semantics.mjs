import assert from "node:assert/strict";
import { cameraReportsForIncident, cameraSemanticsForEvent } from "../src/cameraSemantics.mjs";

const metadataEvent = {
  id: 4,
  created_at: "2026-09-05T12:00:00Z",
  objects: [{
    status: "motion_qualification",
    motion_qualification: {
      camera_semantics: {
        reports: [{
          topic: "tns1:RuleEngine/CellMotionDetector/Motion",
          category: "person",
          reported_class: "Human",
          candidate_model_classes: ["person", "face", "person"],
        }],
      },
    },
  }],
};

assert.deepEqual(cameraSemanticsForEvent(metadataEvent)?.reports?.[0]?.category, "person");
assert.deepEqual(cameraReportsForIncident({ events: [metadataEvent] }), [{
  topic: "tns1:RuleEngine/CellMotionDetector/Motion",
  category: "person",
  reportedClass: "Human",
  candidateModelClasses: ["person", "face"],
  eventId: 4,
  eventAt: "2026-09-05T12:00:00Z",
}]);

const grouped = {
  id: 99,
  events: [
    { id: 9, created_at: "2026-09-05T12:02:00Z", camera_semantics: { reports: [{ topic: "vehicle-topic", category: "vehicle", candidate_model_classes: ["car"] }] } },
    { id: 8, created_at: "2026-09-05T12:01:00Z", camera_semantics: { reports: [{ topic: "animal-topic", category: "animal" }] } },
  ],
};
assert.deepEqual(cameraReportsForIncident(grouped), [
  { topic: "vehicle-topic", category: "vehicle", reportedClass: "", candidateModelClasses: ["car"], eventId: 9, eventAt: "2026-09-05T12:02:00Z" },
  { topic: "animal-topic", category: "animal", reportedClass: "", candidateModelClasses: [], eventId: 8, eventAt: "2026-09-05T12:01:00Z" },
]);

const aggregateOnly = {
  id: 99,
  created_at: "2026-09-05T12:03:00Z",
  camera_semantics: {
    reports: [{
      topic: "face-topic",
      category: "face",
      source_event_id: 5,
      source_created_at: "2026-09-05T12:00:30Z",
      candidate_model_classes: [],
    }],
  },
};
assert.deepEqual(cameraReportsForIncident(aggregateOnly), [{
  topic: "face-topic",
  category: "face",
  reportedClass: "",
  candidateModelClasses: [],
  eventId: 5,
  eventAt: "2026-09-05T12:00:30Z",
}]);
assert.deepEqual(cameraReportsForIncident({ events: [{ id: 7, camera_semantics: { reports: [{ topic: "ignored", category: "unknown" }] } }] }), []);
assert.deepEqual(cameraReportsForIncident({ id: 1, objects: [] }), []);

console.log("camera semantics tests passed");
