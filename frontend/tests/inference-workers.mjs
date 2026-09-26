import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { formatInferenceMs, inferenceTargetRows, workerWeight } from "../src/admin/inferenceWorkers.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const configPage = readFileSync(join(directory, "../src/admin/ConfigPage.jsx"), "utf8");
const workersPanel = readFileSync(join(directory, "../src/admin/InferenceWorkersPanel.jsx"), "utf8");
const constants = readFileSync(join(directory, "../src/shared/constants.js"), "utf8");

assert.equal(formatInferenceMs(35.14), "35.1 ms");
assert.equal(formatInferenceMs(2823), "2823 ms");
assert.equal(formatInferenceMs(null), "—");
assert.equal(workerWeight({ inference_default_worker_weight: 2 }, "worker-a"), 2);
assert.equal(workerWeight({
  inference_default_worker_weight: 2,
  inference_worker_weights: { "worker-a": 0 },
}, "worker-a"), 0);

const rows = inferenceTargetRows({
  loaded_device: "GPU",
  openvino_loaded: true,
  model_load_ms: 240,
  runtime: { total_inferences: 4, failed_inferences: 1, last_inference_ms: 30 },
  isolation: { pending_requests: 2 },
  remote_registry: {
    workers: [{
      worker_id: "worker-a",
      name: "trainer",
      ready: true,
      roles: ["object", "face"],
      pending_requests: 1,
      completed_requests: 8,
      failed_requests: 0,
      lease_remaining_seconds: 12.4,
      statuses: { object: { loaded_device: "GPU", model_load_ms: 350, runtime: { last_inference_ms: 18 } } },
    }],
  },
}, {
  inference_primary_weight: 1,
  inference_default_worker_weight: 3,
});

assert.equal(rows[0].name, "Primary");
assert.equal(rows[0].completed, 4);
assert.equal(rows[0].pending, 2);
assert.equal(rows[1].name, "trainer");
assert.equal(rows[1].device, "GPU");
assert.equal(rows[1].weight, 3);
assert.equal(rows[1].completed, 8);
assert.deepEqual(rows[1].roles, ["object", "face"]);

assert.match(constants, /HEALTH_TELEMETRY_SECTIONS = \["health", "inference", "occupancy"\]/);
assert.match(configPage, /health-tab-inference/);
assert.match(configPage, /InferenceHealthPanel/);
assert.match(configPage, /InferenceWorkersPanel/);
assert.match(configPage, /\["workers", "Workers", Server\]/);
assert.match(workersPanel, /inference_balance/);
assert.match(workersPanel, /inference_primary_weight/);
assert.match(workersPanel, /\/api\/detector\/status/);

console.log("inference worker health and balance ui tests passed");
