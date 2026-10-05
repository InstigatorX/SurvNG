import assert from "node:assert/strict";
import { relatedEvidenceReasons } from "../src/relatedIncidents.mjs";

assert.deepEqual(relatedEvidenceReasons({
  relation_type: "appearance", visually_similar: true, similarity: 0.834,
}), [{ kind: "appearance", label: "Appearance match (ReID) · 83% similarity" }]);
assert.deepEqual(relatedEvidenceReasons({
  relation_type: "sequence_candidate", visually_similar: false, sequence_delta_seconds: 4.2,
}), [{ kind: "time", label: "Nearby in time · 4s apart" }]);
assert.deepEqual(relatedEvidenceReasons({
  relation_type: "expected_route", route_name: "Gate to porch", sequence_delta_seconds: 12,
}).map(({ kind }) => kind), ["route", "time"]);
assert.deepEqual(relatedEvidenceReasons({
  relation_type: "appearance_route", visually_similar: true, similarity: 0.9,
  route_name: "Gate to porch", sequence_delta_seconds: 12,
}).map(({ kind }) => kind), ["appearance", "route", "time"]);
assert.deepEqual(relatedEvidenceReasons({ relation_type: "appearance_sequence" }).map(({ kind }) => kind), ["appearance", "time"]);
assert.deepEqual(relatedEvidenceReasons({ visually_similar: true, similarity: null }), [
  { kind: "appearance", label: "Appearance match (ReID)" },
]);
assert.deepEqual(relatedEvidenceReasons({}), [{ kind: "related", label: "Related incident" }]);
console.log("Related evidence reason tests passed");
