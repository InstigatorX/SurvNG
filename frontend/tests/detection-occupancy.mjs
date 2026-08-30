import assert from "node:assert/strict";
import {
  OCCUPANCY_TONES,
  backupCorrelation,
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
  resolveDetectorHealth,
  resolveObjectWorkerCount,
  siteEffectiveness,
  siteOnvifHealthy,
  trackingHealthVerdict,
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

const keptUpAfterWaits = emaCoverageVerdict({
  coveragePercent: 100,
  deferred: 4362,
  staleSkipped: 1,
  slotCount: 2,
  windowLabel: "last 2 hours",
});
assert.equal(keptUpAfterWaits.tone, OCCUPANCY_TONES.good);
assert.match(keptUpAfterWaits.headline, /100\.00% of visual samples in the last 2 hours/);
assert.match(keptUpAfterWaits.detail, /4,362 times/);
assert.match(keptUpAfterWaits.detail, /since restart/);
assert.match(keptUpAfterWaits.detail, /1 older frame was replaced/);
assert.match(keptUpAfterWaits.detail, /were not dropped/);
assert.match(keptUpAfterWaits.suggestion, /Leave Simultaneous EMA cameras at 2/);

const waitingNowButCovered = emaCoverageVerdict({
  coveragePercent: 100,
  deferred: 4362,
  staleSkipped: 1,
  slotCount: 2,
  analysisWaitP95Ms: 400,
});
assert.equal(waitingNowButCovered.tone, OCCUPANCY_TONES.warning);
assert.match(waitingNowButCovered.headline, /keeping up/);
assert.match(waitingNowButCovered.suggestion, /Leave Simultaneous EMA cameras at 2 unless/);

const longWaitCovered = emaCoverageVerdict({
  coveragePercent: 100,
  deferred: 4362,
  staleSkipped: 1,
  slotCount: 2,
  analysisWaitP95Ms: 1200,
});
assert.equal(longWaitCovered.tone, OCCUPANCY_TONES.warning);
assert.match(longWaitCovered.suggestion, /raise Simultaneous EMA cameras from 2 to 3/);

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
assert.match(halfEma.suggestion, /Wait for camera notice/);
assert.match(halfEma.detail, /1\.5 seconds/);
assert.doesNotMatch(halfEma.detail, /unhealthy/);
assert.equal(halfEma.setting.label, "Wait for camera notice");

const noticesProveOnvif = incidentSplitVerdict({
  cameraObjects: 392,
  emaObjects: 2175,
  backupEnabled: true,
  onvifHealthy: false,
  backupGraceSeconds: 1.5,
});
assert.equal(noticesProveOnvif.tone, OCCUPANCY_TONES.warning);
assert.doesNotMatch(noticesProveOnvif.detail, /unhealthy/);
assert.doesNotMatch(noticesProveOnvif.suggestion, /Fix the camera event connection/);
assert.match(noticesProveOnvif.detail, /event connection is not dead|ONVIF is working/);
assert.match(noticesProveOnvif.detail, /1\.5 seconds/);
assert.match(noticesProveOnvif.suggestion, /Wait for camera notice/);

const racingEma = incidentSplitVerdict({
  cameraObjects: 12,
  emaObjects: 20,
  backupEnabled: true,
  onvifHealthy: true,
  backupMatchedNotices: 40,
  backupWithoutNotices: 8,
  backupGraceSeconds: 1.5,
});
assert.match(racingEma.detail, /waits 1\.5 seconds/);
assert.match(racingEma.detail, /ONVIF is working/);
assert.match(racingEma.suggestion, /Wait for camera notice/);

const brokenOnvif = incidentSplitVerdict({ cameraObjects: 0, emaObjects: 18, backupEnabled: true, onvifHealthy: false });
assert.equal(brokenOnvif.tone, OCCUPANCY_TONES.warning);
assert.match(brokenOnvif.suggestion, /Fix the camera event connection/);
assert.match(brokenOnvif.detail, /no camera notices arrived/);

assert.equal(cameraOnvifHealthy({ onvif: { enabled: false, connected: false } }), null);
assert.equal(cameraOnvifHealthy({ onvif: { enabled: true, connected: false } }), false);
assert.equal(cameraOnvifHealthy({ onvif: { enabled: true, connected: true } }), true);
assert.equal(siteOnvifHealthy([
  { onvif: { enabled: false, connected: false } },
  { onvif: { enabled: true, connected: true } },
]), true);
assert.equal(siteOnvifHealthy([
  { onvif: { enabled: false, connected: false } },
  { onvif: { enabled: true, connected: false } },
]), false);
assert.deepEqual(backupCorrelation([
  { onvif: { ema_onvif_matches: 4, ema_without_onvif: 1 } },
  { motion: { visual_backup_onvif_matches: 2 } },
]), { matches: 6, without: 1 });

assert.equal(visualBackupVerdict({}).tone, OCCUPANCY_TONES.idle);
assert.equal(visualBackupVerdict({ attempts: 6, objects: 3, none: 3 }).tone, OCCUPANCY_TONES.good);
assert.equal(visualBackupVerdict({ attempts: 6, objects: 0, none: 6 }).tone, OCCUPANCY_TONES.warning);

assert.equal(detectorLaneVerdict({ trackingEnabled: true, workerCount: 2 }).tone, OCCUPANCY_TONES.good);
assert.equal(detectorLaneVerdict({ trackingEnabled: true, workerCount: 1 }).tone, OCCUPANCY_TONES.bad);
assert.match(detectorLaneVerdict({ trackingEnabled: true, workerCount: 1 }).suggestion, /Set Parallel detectors to 2/);
assert.equal(detectorLaneVerdict({ trackingEnabled: false, workerCount: 1 }).tone, OCCUPANCY_TONES.good);
assert.equal(detectorLaneVerdict({ backend: "coreml", workerCount: 1 }).tone, OCCUPANCY_TONES.good);

const trackingLane = detectorLaneVerdict({ trackingEnabled: true, workerCount: 2 });
assert.equal(trackingLane.setting.detectionSection, "object");
assert.equal(trackingLane.setting.label, "Parallel detectors");
assert.equal(detectorLaneVerdict({ trackingEnabled: false, workerCount: 3 }).setting.detectionSection, "object");

const threeDetectors = detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 3,
  runningWorkerCount: 3,
});
assert.equal(threeDetectors.tone, OCCUPANCY_TONES.good);
assert.match(threeDetectors.headline, /3 detectors/);
assert.equal(threeDetectors.setting.detectionSection, "object");

const staleWorkers = detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 3,
  runningWorkerCount: 2,
});
assert.equal(staleWorkers.tone, OCCUPANCY_TONES.warning);
assert.match(staleWorkers.headline, /2 detectors running/);
assert.match(staleWorkers.headline, /Parallel detectors is 3/);
assert.equal(staleWorkers.setting.detectionSection, "object");
assert.equal(staleWorkers.setting.label, "Parallel detectors");

assert.equal(detectorLaneVerdict({ enabled: false, workerCount: 2 }).tone, OCCUPANCY_TONES.idle);
assert.equal(detectorLaneVerdict({ loaded: false, workerCount: 2 }).tone, OCCUPANCY_TONES.bad);
assert.equal(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 3,
  runningWorkerCount: 0,
}).tone, OCCUPANCY_TONES.bad);
assert.match(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 3,
  runningWorkerCount: 0,
}).headline, /No detector process is running/);
assert.equal(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 2,
  runningWorkerCount: 2,
  fallbackActive: true,
}).tone, OCCUPANCY_TONES.warning);
assert.match(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 2,
  runningWorkerCount: 2,
  fallbackActive: true,
}).headline, /CPU fallback/);
assert.equal(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 2,
  runningWorkerCount: 2,
  pending: 3,
}).tone, OCCUPANCY_TONES.warning);
assert.equal(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 2,
  runningWorkerCount: 2,
  failed: 4,
}).tone, OCCUPANCY_TONES.warning);
assert.equal(detectorLaneVerdict({
  trackingEnabled: true,
  configuredWorkerCount: 2,
  runningWorkerCount: 2,
  crashes: 1,
}).tone, OCCUPANCY_TONES.warning);

const liveHealth = resolveDetectorHealth({
  config: { detector: { enabled: true, object_worker_count: 3, tracking: { enabled: true }, backend: "openvino" } },
  telemetry: {
    detector: {
      enabled: true,
      loaded_backend: "openvino",
      loaded_device: "GPU",
      workers: { object: { configured_workers: 3, alive_workers: 3, pending_requests: 0, crash_count: 0 } },
      runtime: { failed_inferences: 0, queue_depth: 0 },
    },
  },
});
assert.equal(liveHealth.configured, 3);
assert.equal(liveHealth.running, 3);
assert.equal(liveHealth.loaded, true);
assert.equal(detectorLaneVerdict(liveHealth).tone, OCCUPANCY_TONES.good);

const fromConfigAndLive = resolveObjectWorkerCount({
  config: { detector: { object_worker_count: 3 } },
  telemetry: { detector: { workers: { object: { configured_workers: 2, alive_workers: 2 } } } },
});
assert.equal(fromConfigAndLive.configured, 3);
assert.equal(fromConfigAndLive.running, 2);

const fromIsolation = resolveObjectWorkerCount({
  telemetry: { detector: { isolation: { configured_workers: 3, alive_workers: 3 } } },
});
assert.equal(fromIsolation.configured, 3);
assert.equal(fromIsolation.running, 3);

const oneDown = resolveObjectWorkerCount({
  config: { detector: { object_worker_count: 3 } },
  telemetry: { detector: { workers: { object: { configured_workers: 3, alive_workers: 2 } } } },
});
assert.equal(oneDown.configured, 3);
assert.equal(oneDown.running, 2);

const history = coverageFromRuntimeHistory([
  { analysis_frames_sampled: 90, analysis_frames_dropped: 10 },
  { analysis_frames_sampled: 100, analysis_frames_dropped: 0 },
]);
assert.equal(history.coveragePercent, 95);
assert.equal(history.staleSkipped, 10);
assert.equal(history.windowLabel, "last 2 hours");

const live = coverageFromCameraMotion({
  motion: {
    analysis_frames_dropped: 4,
    analysis_wait_ms_p95: 80,
    analysis_runtime: { frames_sampled: 16, analysis_slot_deferrals: 3, capture_to_analysis_p95_ms: 40 },
  },
});
assert.equal(live.coveragePercent, 80);
assert.equal(live.deferred, 3);
assert.equal(live.analysisWaitP95Ms, 80);

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
assert.equal(report.summary.headline, "Detection looks healthy");
assert.match(report.summary.detail, /admission/i);
assert.match(report.summary.detail, /tracking/i);
assert.match(report.summary.detail, /worker capacity/i);
assert.match(report.summary.detail, /visual analysis/i);
assert.deepEqual(report.rows.map((row) => row.id), [
  "coverage",
  "incident-split",
  "visual-backup",
  "double-check",
  "detector-lane",
  "eligibility",
]);
assert.deepEqual(report.pillars.map((row) => row.id), [
  "admission",
  "engine",
  "tracking",
  "capacity",
]);
const reportLane = report.rows.find((row) => row.id === "detector-lane");
assert.equal(reportLane.setting.detectionSection, "object");
assert.equal(reportLane.setting.label, "Parallel detectors");
assert.match(report.pillars.find((row) => row.id === "engine").headline, /2 detectors/);
assert.equal(report.pillars.find((row) => row.id === "tracking").tone, OCCUPANCY_TONES.good);
assert.match(report.context, /Only objects inside a zone/);

const mismatchedReport = buildOccupancyReport({
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
  workerCount: 3,
  configuredWorkerCount: 3,
  runningWorkerCount: 2,
  backend: "openvino",
  requireZone: true,
  backupEnabled: true,
  onvifHealthy: true,
});
const mismatchedLane = mismatchedReport.rows.find((row) => row.id === "detector-lane");
assert.equal(mismatchedLane.tone, OCCUPANCY_TONES.warning);
assert.equal(mismatchedReport.summary.headline, "Detection needs a look");
assert.equal(mismatchedLane.setting.detectionSection, "object");
assert.match(mismatchedLane.headline, /Parallel detectors is 3/);

assert.equal(trackingHealthVerdict({ trackingEnabled: false }).tone, OCCUPANCY_TONES.idle);
assert.equal(trackingHealthVerdict({ trackingEnabled: true, workerCount: 2 }).tone, OCCUPANCY_TONES.good);
assert.equal(trackingHealthVerdict({ trackingEnabled: true, workerCount: 1 }).tone, OCCUPANCY_TONES.bad);
assert.equal(trackingHealthVerdict({
  trackingEnabled: true,
  workerCount: 2,
  trackingSkipped: 4,
  trackingAttempts: 12,
}).tone, OCCUPANCY_TONES.warning);
assert.equal(trackingHealthVerdict({ trackingEnabled: true, backend: "coreml", workerCount: 1 }).tone, OCCUPANCY_TONES.good);

const trackingShort = buildOccupancyReport({
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
  workerCount: 1,
  configuredWorkerCount: 1,
  runningWorkerCount: 1,
  backend: "openvino",
  requireZone: true,
  backupEnabled: true,
  onvifHealthy: true,
});
assert.equal(trackingShort.tone, OCCUPANCY_TONES.bad);
assert.equal(trackingShort.summary.headline, "Detection needs attention");
assert.match(trackingShort.summary.detail, /only one detector/);
assert.equal(trackingShort.pillars.find((row) => row.id === "engine").tone, OCCUPANCY_TONES.good);
assert.equal(trackingShort.pillars.find((row) => row.id === "tracking").tone, OCCUPANCY_TONES.bad);

const highEmaHealthyEngine = buildOccupancyReport({
  coverage: { coveragePercent: 99, deferred: 0, staleSkipped: 0 },
  effectiveness: {
    camera_object_events: 10,
    ema_object_events: 10,
    visual_backup_attempts: 8,
    visual_backup_objects: 6,
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
assert.equal(highEmaHealthyEngine.tone, OCCUPANCY_TONES.warning);
assert.equal(highEmaHealthyEngine.summary.headline, "Detection needs a look");
assert.equal(highEmaHealthyEngine.pillars.find((row) => row.id === "engine").tone, OCCUPANCY_TONES.good);
assert.equal(highEmaHealthyEngine.pillars.find((row) => row.id === "admission").tone, OCCUPANCY_TONES.warning);
assert.match(highEmaHealthyEngine.summary.detail, /visual backup/i);
assert.doesNotMatch(highEmaHealthyEngine.pillars.find((row) => row.id === "admission").detail, /unhealthy/);
assert.match(highEmaHealthyEngine.pillars.find((row) => row.id === "admission").suggestion, /Wait for camera notice/);

const onvifWorkingHighEma = buildOccupancyReport({
  coverage: { coveragePercent: 99, deferred: 0, staleSkipped: 0 },
  effectiveness: {
    camera_object_events: 392,
    ema_object_events: 2175,
    visual_backup_attempts: 2175,
    visual_backup_objects: 2175,
    suppression_verification_checks: 0,
    suppression_verification_rescues: 0,
  },
  slotCount: 2,
  trackingEnabled: true,
  workerCount: 2,
  backend: "openvino",
  requireZone: true,
  backupEnabled: true,
  onvifHealthy: false,
  backupGraceSeconds: 1.5,
});
const onvifWorkingAdmission = onvifWorkingHighEma.pillars.find((row) => row.id === "admission");
assert.equal(onvifWorkingAdmission.tone, OCCUPANCY_TONES.warning);
assert.doesNotMatch(onvifWorkingAdmission.detail, /unhealthy/);
assert.doesNotMatch(onvifWorkingAdmission.suggestion, /Fix the camera event connection/);
assert.match(onvifWorkingAdmission.detail, /1\.5 seconds/);
assert.match(onvifWorkingAdmission.suggestion, /Wait for camera notice/);

const wastedChecks = buildOccupancyReport({
  coverage: { coveragePercent: 99, deferred: 0, staleSkipped: 0 },
  effectiveness: {
    camera_object_events: 12,
    ema_object_events: 2,
    visual_backup_attempts: 2,
    visual_backup_objects: 2,
    suppression_verification_checks: 80,
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
assert.ok(wastedChecks.pillars.some((row) => row.id === "waste"));
assert.equal(wastedChecks.pillars.find((row) => row.id === "waste").tone, OCCUPANCY_TONES.bad);

const cameraOnly = buildOccupancyReport({
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
  workerCount: 1,
  backend: "openvino",
  requireZone: true,
  backupEnabled: true,
  onvifHealthy: true,
  includeDetectorHealth: false,
});
assert.deepEqual(cameraOnly.pillars.map((row) => row.id), ["admission", "capacity"]);
assert.equal(cameraOnly.tone, OCCUPANCY_TONES.good);

console.log("detection occupancy helpers passed");
