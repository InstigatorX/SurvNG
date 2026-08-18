import assert from "node:assert/strict";
import { relatedEvidenceLabel, relatedIncidentThumbnailPath, relatedIncidentsPath, visibleRelatedAppearances } from "../src/relatedIncidents.mjs";

assert.equal(relatedIncidentsPath(42), "/api/events/42/related-incidents?hours=24&limit=16");
assert.equal(relatedIncidentsPath(42, 48, 8), "/api/events/42/related-incidents?hours=48&limit=8");
assert.equal(relatedIncidentThumbnailPath(42), "/api/events/42/thumbnail.jpg?width=360&quality=80");
assert.equal(relatedIncidentThumbnailPath(42, 720, 92), "/api/events/42/thumbnail.jpg?width=720&quality=92");

const matches = visibleRelatedAppearances({
  matches: [
    { event_id: 7, similarity: 1, visually_similar: true },
    { event_id: 8, similarity: 0.82, visually_similar: true },
    { event_id: 9, similarity: 0.95, visually_similar: false },
    { event_id: 11, sequence_delta_seconds: 4.2, relation_type: "sequence_candidate", visually_similar: false },
    { event_id: 12, sequence_delta_seconds: 3.8, relation_type: "expected_route", route_name: "Back yard to gate", visually_similar: false },
    { event_id: 10, similarity: 0.91, visually_similar: true },
    { event_id: 8, similarity: 0.89, visually_similar: true },
    { event_id: "invalid", similarity: 0.99, visually_similar: true },
  ]
}, 7);

assert.deepEqual(matches.map((match) => match.event_id), [12, 10, 8, 11]);
assert.deepEqual(visibleRelatedAppearances({ matches: [] }, 7), []);
assert.equal(visibleRelatedAppearances({ matches: Array.from({ length: 20 }, (_, index) => ({ event_id: index + 1, similarity: 1 - index / 100, visually_similar: true })) }, 99, 3).length, 3);
assert.equal(relatedEvidenceLabel({ similarity: 0.834, visually_similar: true, sequence_delta_seconds: 4.2 }), "Appearance 83% · 4s");
assert.equal(relatedEvidenceLabel({ sequence_delta_seconds: 4.2 }), "Likely · 4s");
assert.equal(relatedEvidenceLabel({ relation_type: "expected_route", sequence_delta_seconds: 4.2 }), "Expected · 4s");
assert.equal(relatedEvidenceLabel({ relation_type: "appearance_route", sequence_delta_seconds: 4.2, similarity: 0.834, visually_similar: true }), "Expected · Appearance 83% · 4s");

console.log("related incident tests passed");
