import assert from "node:assert/strict";
import {
  assistantActiveExportIds,
  cacheExportJobs,
  cachedExportJob,
  fetchAssistantExportJobs,
  fetchExportJob,
  mergeAssistantExportJobs,
  resetExportPollingCacheForTests,
} from "../src/exportPolling.mjs";

const messages = [{
  id: "answer",
  evidence: [
    { id: "one", details: { media_export: { id: "job-b", status: "running", progress: 10 } } },
    { id: "two", details: { media_export: { id: "job-a", status: "queued", progress: 0 } } },
    { id: "duplicate", details: { media_export: { id: "job-a", status: "queued", progress: 0 } } },
    { id: "done", details: { media_export: { id: "job-c", status: "completed", progress: 100 } } },
  ],
}];
assert.deepEqual(assistantActiveExportIds(messages), ["job-a", "job-b"]);

const merged = mergeAssistantExportJobs(messages, [{
  id: "job-a", status: "running", phase: "Encoding", progress: 55, download_url: "",
}]);
assert.equal(merged[0].evidence[1].details.media_export.progress, 55);
assert.equal(merged[0].evidence[2].details.media_export.phase, "Encoding");
assert.equal(merged[0].evidence[0].details.media_export.progress, 10);

resetExportPollingCacheForTests();
cacheExportJobs([{ id: "cached", status: "running" }], 1_000);
assert.equal(cachedExportJob("cached", 500, 1_400)?.status, "running");
assert.equal(cachedExportJob("cached", 500, 1_501), null);

resetExportPollingCacheForTests();
let detailRequests = 0;
let releaseDetail;
const delayedRequest = async () => {
  detailRequests += 1;
  await new Promise((resolve) => { releaseDetail = resolve; });
  return { ok: true, status: 200, json: async () => ({ id: "same", status: "running" }) };
};
const first = fetchExportJob("same", delayedRequest);
const second = fetchExportJob("same", delayedRequest);
await Promise.resolve();
releaseDetail();
assert.equal((await first).id, "same");
assert.equal((await second).id, "same");
assert.equal(detailRequests, 1);

resetExportPollingCacheForTests();
const paths = [];
const batchRequest = async (path) => {
  paths.push(path);
  if (path.startsWith("/api/exports?")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        exports: [
          { id: "job-a", status: "running" },
          { id: "job-c", status: "queued" },
        ],
      }),
    };
  }
  return { ok: true, status: 200, json: async () => ({ id: "job-b", status: "completed" }) };
};
const jobs = await fetchAssistantExportJobs(["job-a", "job-b", "job-c"], batchRequest);
assert.deepEqual(jobs.map((job) => job.id).sort(), ["job-a", "job-b", "job-c"]);
assert.deepEqual(paths, [
  "/api/exports?status=active&limit=500",
  "/api/exports/job-b",
]);

console.log("export polling tests passed");
