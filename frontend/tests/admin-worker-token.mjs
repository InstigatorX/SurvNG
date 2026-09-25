import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(directory, "../src/admin/ConfigPage.jsx"),
  "utf8",
);

assert.match(
  source,
  /fetch\("\/api\/config\/inference-worker-token", \{ method: "POST" \}\)/,
);
assert.match(
  source,
  /fetch\("\/api\/config\/inference-worker-token", \{ method: "DELETE" \}\)/,
);
assert.match(
  source,
  /commitImmediateConfig\(\["inference_workers", "worker_token_hash"\], "__SURVNG_SECRET_SET__"\)/,
);
assert.match(
  source,
  /Rotate the inference worker token\? Existing workers stay connected/,
);
assert.match(source, /SURVNG_INFERENCE_TOKEN/);
assert.doesNotMatch(
  source,
  /The new API token secret is shown only once/,
);

console.log("inference worker token settings tests passed");
