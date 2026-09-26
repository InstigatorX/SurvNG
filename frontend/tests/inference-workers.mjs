import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { formatAttemptOutcome, formatCommit, formatInferenceMs, formatRoleAttempts, inferenceTargetRows, workerWeight } from "../src/admin/inferenceWorkers.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const configPage = readFileSync(join(directory, "../src/admin/ConfigPage.jsx"), "utf8");
const workersPanel = readFileSync(join(directory, "../src/admin/InferenceWorkersPanel.jsx"), "utf8");
const healthPanel = readFileSync(join(directory, "../src/admin/InferenceHealthPanel.jsx"), "utf8");
const constants = readFileSync(join(directory, "../src/shared/constants.js"), "utf8");

assert.equal(formatCommit("abcdef1234567890abcdef1234567890abcdef12"), "abcdef123456");
assert.equal(formatCommit(""), "—");
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
      rerouted_requests: 3,
      failed_requests: 1,
      last_outcome: "rerouted",
      last_error: "remote reid embed_person timed out",
      last_role: "reid",
      last_operation: "embed_person",
      last_inference_ms: 22,
      last_request_ms: 40,
      average_inference_ms: 19.5,
      role_attempts: { reid: { completed: 2, rerouted: 3, failed: 0 }, object: { completed: 6, rerouted: 0, failed: 1 } },
      software_version: "abcdef1234567890abcdef1234567890abcdef12",
      upgrade_phase: "failed",
      upgrade_detail: "survng-inference-upgrade.path is not enabled on this worker",
      routing_hold: true,
      lease_remaining_seconds: 12.4,
      statuses: { object: { loaded_device: "GPU", model_load_ms: 350, runtime: { last_inference_ms: 18 } } },
    }],
  },
  primary_sha: "1234567890abcdef1234567890abcdef12345678",
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
assert.equal(rows[1].rerouted, 3);
assert.equal(rows[1].failed, 1);
assert.equal(rows[1].lastInferenceMs, 22);
assert.equal(rows[1].lastRequestMs, 40);
assert.equal(rows[1].softwareVersion, "abcdef1234567890abcdef1234567890abcdef12");
assert.equal(rows[1].codeMatches, false);
assert.equal(rows[1].upgradePhase, "failed");
assert.equal(rows[1].routingHold, true);
assert.equal(rows[1].averageInferenceMs, 19.5);
assert.equal(rows[0].rerouted, null);
assert.equal(
  formatAttemptOutcome(rows[1]),
  "Rerouted reid embed_person: remote reid embed_person timed out",
);
assert.match(formatRoleAttempts(rows[1].roleAttempts), /reid: 2 completed, 3 rerouted/);
assert.deepEqual(rows[1].roles, ["object", "face"]);

assert.match(constants, /HEALTH_TELEMETRY_SECTIONS = \["health", "inference", "occupancy"\]/);
assert.match(configPage, /health-tab-inference/);
assert.match(configPage, /InferenceHealthPanel/);
assert.match(configPage, /InferenceWorkersPanel/);
assert.match(configPage, /\["workers", "Workers", Server\]/);
assert.match(workersPanel, /Rerouted/);
assert.match(workersPanel, /Match primary code/);
assert.match(workersPanel, /Paused until its queue finishes/);
assert.match(healthPanel, /Paused until its queue finishes/);
assert.match(workersPanel, /Primary code/);
assert.match(workersPanel, /upgradeError/);
assert.match(workersPanel, /\/api\/inference\/workers\//);
assert.match(workersPanel, /inference_balance/);
assert.match(workersPanel, /inference_primary_weight/);
assert.match(workersPanel, /\/api\/detector\/status/);

console.log("inference worker health and balance ui tests passed");
