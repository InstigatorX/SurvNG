import assert from "node:assert/strict";
import {
  OCCUPANCY_TONES,
  buildOccupancyReport,
  cameraEffectiveness,
  cameraOnvifHealthy,
  coverageFromCameraMotion,
  coverageFromRuntimeHistory,
  detectorLaneVerdict,
  doubleCheckVerdict,
  emaCoverageVerdict,
  incidentSplitVerdict,
  mergeEffectiveness,
  siteEffectiveness,
  siteOnvifHealthy,
  visualBackupVerdict,
  worstOccupancyTone,
} from "../src/detectionOccupancy.mjs";

assert.equal(worstOccupancyTone(["idle", "good", "warning"]), OCCUPANCY_TONES.warning);
assert.equal(worstOccupancyTone(["good", "bad", "warning"]), OCCUPANCY_TONES.bad);

const idleCoverage = emaCoverageVerdict({});
assert.equal(idleCoverage.tone, OCCUPANCY_TONES.idle);
assert.match(idleCoverage.suggestion, /Leave Simultaneous EMA cameras/);

const healthyCoverage = emaCoverageVerdict({
  coveragePercent: 99.4,
  deferred: 0,
  staleSkipped: 12,
  slotCount: 2,
});
assert.equal(healthyCoverage.tone, OCCUPANCY_TONES.good);
assert.match(healthyCoverage.suggestion, /Leave Simultaneous EMA cameras at 2/);

const waitingCoverage = emaCoverageVerdict({
  coveragePercent: 91,
  deferred: 18,
  staleSkipped: 40,
  slotCount: 2,
});
assert.equal(waitingCoverage.tone, OCCUPANCY_TONES.warning);
assert.match(waitingCoverage.suggestion, /Raise Simultaneous EMA cameras from 2 to 3/);

const behindCoverage = emaCoverageVerdict({
  coveragePercent: 72,
  deferred: 80,
  slotCount: 2,
});
assert.equal(behindCoverage.tone, OCCUPANCY_TONES.bad);
assert.match(behindCoverage.suggestion, /Raise Simultaneous EMA cameras from 2 to 3/);

const streamIssue = emaCoverageVerdict({
  coveragePercent: 88,
  deferred: 0,
  staleSkipped: 30,
});
assert.equal(streamIssue.tone, OCCUPANCY_TONES.warning);
assert.match(streamIssue.suggestion, /Do not raise Simultaneous EMA cameras/);

assert.equal(doubleCheckVerdict({}).tone, OCCUPANCY_TONES.good);
assert.match(doubleCheckVerdict({}).suggestion, /Leave Double-check filtered motion/);
assert.equal(doubleCheckVerdict({ checks: 8, rescues: 2 }).tone, OCCUPANCY_TONES.good);
assert.equal(doubleCheckVerdict({ checks: 6, rescues: 0 }).tone, OCCUPANCY_TONES.idle);
assert.equal(doubleCheckVerdict({ checks: 24, rescues: 0 }).tone, OCCUPANCY_TONES.warning);
assert.match(doubleCheckVerdict({ checks: 24, rescues: 0 }).suggestion, /About 1 in 100/);
assert.equal(doubleCheckVerdict({ checks: 80, rescues: 0 }).tone, OCCUPANCY_TONES.bad);

const mostlyCamera = incidentSplitVerdict({ cameraObjects: 20, emaObjects: 4, backupEnabled: true, onvifHealthy: true });
assert.equal(mostlyCamera.tone, OCCUPANCY_TONES.good);
assert.match(mostlyCamera.suggestion, /Keep Camera \+ EMA backup/);

const halfEma = incidentSplitVerdict({ cameraObjects: 10, emaObjects: 10, backupEnabled: true, onvifHealthy: true });
assert.equal(halfEma.tone, OCCUPANCY_TONES.warning);
assert.match(halfEma.suggestion, /Keep Camera \+ EMA backup/);
assert.match(halfEma.suggestion, /Do not switch to camera notices only/);
assert.match(halfEma.detail, /not a looser object detector/);

const racingEma = incidentSplitVerdict({
  cameraObjects: 398,
  emaObjects: 2180,
  backupEnabled: true,
  onvifHealthy: false,
  backupMatchedNotices: 40,
  backupWithoutNotices: 5,
});
assert.equal(racingEma.tone, OCCUPANCY_TONES.warning);
assert.match(racingEma.detail, /1\.5 seconds/);
assert.doesNotMatch(racingEma.detail, /unhealthy/);
assert.match(racingEma.suggestion, /Wait for camera notice/);

const brokenOnvif = incidentSplitVerdict({ cameraObjects: 0, emaObjects: 18, backupEnabled: true, onvifHealthy: false });
assert.equal(brokenOnvif.tone, OCCUPANCY_TONES.warning);
assert.match(brokenOnvif.suggestion, /Fix the camera event connection/);

assert.equal(visualBackupVerdict({}).tone, OCCUPANCY_TONES.idle);
assert.equal(visualBackupVerdict({ attempts: 6, objects: 3, none: 3 }).tone, OCCUPANCY_TONES.good);
assert.equal(visualBackupVerdict({ attempts: 6, objects: 0, none: 6 }).tone, OCCUPANCY_TONES.warning);

assert.equal(detectorLaneVerdict({ trackingEnabled: true, workerCount: 2 }).tone, OCCUPANCY_TONES.good);
assert.equal(detectorLaneVerdict({ trackingEnabled: true, workerCount: 1 }).tone, OCCUPANCY_TONES.bad);
assert.match(detectorLaneVerdict({ trackingEnabled: true, workerCount: 1 }).suggestion, /Set Parallel detectors to 2/);
assert.equal(detectorLaneVerdict({ trackingEnabled: false, workerCount: 1 }).tone, OCCUPANCY_TONES.good);
assert.equal(detectorLaneVerdict({ backend: "coreml", workerCount: 1 }).tone, OCCUPANCY_TONES.good);

const history = coverageFromRuntimeHistory([
  { analysis_frames_sampled: 90, analysis_frames_dropped: 10 },
  { analysis_frames_sampled: 100, analysis_frames_dropped: 0 },
]);
assert.equal(history.coveragePercent, 95);
assert.equal(history.staleSkipped, 10);

const live = coverageFromCameraMotion({
  motion: {
    analysis_frames_dropped: 4,
    analysis_runtime: { frames_sampled: 16, analysis_slot_deferrals: 3, capture_to_analysis_p95_ms: 40 },
  },
});
assert.equal(live.coveragePercent, 80);
assert.equal(live.deferred, 3);

const merged = mergeEffectiveness([
  { object_events: 4, camera_object_events: 3, ema_object_events: 1, suppression_verification_checks: 2 },
  { object_events: 2, camera_object_events: 0, ema_object_events: 2, suppression_verification_rescues: 1 },
]);
assert.equal(merged.object_events, 6);
assert.equal(merged.camera_object_events, 3);
assert.equal(merged.ema_object_events, 3);
assert.equal(merged.suppression_verification_checks, 2);
assert.equal(merged.suppression_verification_rescues, 1);

const fallback = mergeEffectiveness([
  { object_events: 5, visual_backup_objects: 2 },
]);
assert.equal(fallback.ema_object_events, 2);
assert.equal(fallback.camera_object_events, 3);

const byCamera = {
  porch: { camera_rescue: { object_events: 2, camera_object_events: 1, ema_object_events: 1 } },
  drive: { camera: { object_events: 3, camera_object_events: 3, ema_object_events: 0 } },
};
assert.equal(cameraEffectiveness(byCamera, "porch", "camera_rescue").ema_object_events, 1);
assert.equal(siteEffectiveness(byCamera, [{ id: "porch" }, { id: "drive" }]).camera_object_events, 4);

const report = buildOccupancyReport({
  coverage: { coveragePercent: 99, deferred: 0, staleSkipped: 0 },
  effectiveness: {
    camera_object_events: 12,
    ema_object_events: 2,
    visual_backup_attempts: 2,
    visual_backup_objects: 2,
    suppression_verification_checks: 0,
    suppression_verification_rescues: 0,
  },
  slotCount: 2,
  trackingEnabled: true,
  workerCount: 2,
  backend: "openvino",
  requireZone: true,
  backupEnabled: true,
  onvifHealthy: true,
});
assert.equal(report.tone, OCCUPANCY_TONES.good);
assert.deepEqual(report.rows.map((row) => row.id), [
  "coverage",
  "incident-split",
  "visual-backup",
  "double-check",
  "detector-lane",
  "eligibility",
]);

assert.equal(cameraOnvifHealthy({ onvif: { enabled: false, connected: false } }), null);
assert.equal(cameraOnvifHealthy({ onvif: { enabled: true, connected: false } }), false);
assert.equal(cameraOnvifHealthy({ onvif: { enabled: true, connected: true } }), true);
assert.equal(siteOnvifHealthy([
  { onvif: { enabled: true, connected: true } },
  { onvif: { enabled: true, connected: false } },
  { onvif: { enabled: false, connected: false } },
]), true);
assert.equal(siteOnvifHealthy([
  { onvif: { enabled: true, connected: false } },
  { onvif: { enabled: false, connected: false } },
]), false);

console.log("detection occupancy helpers passed");
