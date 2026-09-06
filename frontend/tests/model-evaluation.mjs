import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { modelEvaluationCoverage } from "../src/modelEvaluation.mjs";

const result = (compared, failed, baselineErrors, candidateErrors) => ({
  sample_count: 10,
  compared_sample_count: compared,
  failed_sample_count: failed,
  baseline: { errors: baselineErrors },
  candidate: { errors: candidateErrors },
  disagreements: [],
});

assert.equal(modelEvaluationCoverage(null), null);
assert.equal(modelEvaluationCoverage(result(10, 0, 0, 0)).complete, true);
assert.deepEqual(modelEvaluationCoverage(result(0, 10, 10, 10)), {
  total: 10, compared: 0, failed: 10, baselineErrors: 10, candidateErrors: 10, complete: false,
});
assert.equal(modelEvaluationCoverage(result(0, 10, 0, 10)).complete, false);
const partial = modelEvaluationCoverage(result(7, 3, 1, 2));
assert.equal(partial.compared, 7);
assert.equal(partial.failed, 3);
assert.equal(partial.complete, false, "agreement on seven images must not claim agreement on the full corpus");

// Older job payloads have only per-model errors; never promote those to success.
assert.equal(modelEvaluationCoverage({ sample_count: 10, baseline: { errors: 10 }, candidate: { errors: 10 } }).complete, false);
assert.equal(modelEvaluationCoverage({ sample_count: 10, baseline: { errors: 0 }, candidate: { errors: 0 } }).complete, true);

const panel = readFileSync(new URL("../src/admin/ModelsAndHardwarePanel.jsx", import.meta.url), "utf8");
assert.match(panel, /evaluationCoverage\.complete \? <div className="probe-result ok"><strong>No label disagreements/);
assert.match(panel, /"Partial comparison" : "No valid comparison"/);
assert.match(panel, /Baseline errors: \{evaluationCoverage\.baselineErrors\}; candidate errors: \{evaluationCoverage\.candidateErrors\}/);
assert.match(panel, /No label disagreements among successful comparisons/);
console.log("model evaluation failure and partial-result tests passed");
