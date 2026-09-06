import assert from "node:assert/strict";
import {
  TRACKING_SAMPLING_PROFILES,
  successfulTrackingComparisonEngines,
  trackingComparisonEngines,
  trackingComparisonRequestUrl,
  trackingComparisonResultsArtifact,
  trackingEngineLabel,
  trackingHistoryVerdictLabel,
  trackingHistorySummaryEntries,
  trackingVerdictLabel,
} from "../src/trackingComparison.mjs";

assert.deepEqual(TRACKING_SAMPLING_PROFILES.map((profile) => profile.value), ["fixed_2fps", "fixed_075fps", "sparse_gaps"]);
assert.equal(trackingComparisonRequestUrl(12, "sparse_gaps"), "/api/events/12/tracking-comparison?duration_seconds=30&sampling_profile=sparse_gaps");
assert.deepEqual(trackingComparisonResultsArtifact({ replay: { frames: [1] }, replay_id: "source", engines: { hybrid: { frame_observations: [] } } }), {
  replay_id: "source", engines: { hybrid: { frame_observations: [] } },
});
assert.equal(trackingEngineLabel("survng_hybrid"), "Hybrid (current)");
assert.equal(trackingEngineLabel("survng_hybrid_candidate"), "Hybrid (candidate)");
assert.equal(trackingEngineLabel("ultralytics_tracktrack"), "TrackTrack");
assert.equal(trackingEngineLabel("ultralytics_botsort"), "BoT-SORT");
assert.equal(trackingEngineLabel("ultralytics_fasttrack"), "FastTrack (legacy)");
assert.equal(trackingVerdictLabel("ultralytics_botsort"), "BoT-SORT");
assert.equal(trackingHistoryVerdictLabel("ultralytics_botsort"), "BoT-SORT (historic)");
assert.equal(trackingHistoryVerdictLabel("ultralytics_botsort", { replay_id: "new-replay" }), "BoT-SORT");
assert.deepEqual(trackingHistorySummaryEntries({ survng_hybrid: 2, ultralytics_botsort: 1, inconclusive: 3, unreviewed: 4 }, [
  { verdict: "ultralytics_botsort", result: { replay_id: "new-replay" } },
]), [
  { verdict: "survng_hybrid", count: 2, label: "Hybrid (current) better" },
  { verdict: "ultralytics_botsort", count: 1, label: "BoT-SORT better" },
  { verdict: "inconclusive", count: 3, label: "No clear winner" },
]);
const comparison = { engines: { survng_hybrid: {}, ultralytics_tracktrack: { error: "Unavailable" } } };
assert.equal(trackingComparisonEngines(comparison).length, 2);
assert.deepEqual(successfulTrackingComparisonEngines(comparison).map(([implementation]) => implementation), ["survng_hybrid"]);

console.log("tracking comparison UI helpers passed");
