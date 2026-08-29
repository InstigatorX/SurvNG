export const OCCUPANCY_TONES = Object.freeze({
  good: "good",
  warning: "warning",
  bad: "bad",
  idle: "idle",
});

export const EMA_TRIGGER_SOURCES = Object.freeze([
  "adaptive",
  "visual_backup",
  "adaptive/visual_backup",
  "ema",
]);

const TONE_RANK = Object.freeze({
  [OCCUPANCY_TONES.idle]: 0,
  [OCCUPANCY_TONES.good]: 1,
  [OCCUPANCY_TONES.warning]: 2,
  [OCCUPANCY_TONES.bad]: 3,
});

export function asCount(value) {
  const count = Number(value);
  return Number.isFinite(count) && count > 0 ? Math.round(count) : 0;
}

export function asPercent(value) {
  const percent = Number(value);
  return Number.isFinite(percent) ? percent : null;
}

export function isEmaTriggerSource(value) {
  return EMA_TRIGGER_SOURCES.includes(String(value || "").trim().toLowerCase());
}

export function worstOccupancyTone(tones) {
  return (tones || []).reduce((current, tone) => (
    (TONE_RANK[tone] || 0) > (TONE_RANK[current] || 0) ? tone : current
  ), OCCUPANCY_TONES.idle);
}

export function mergeEffectiveness(summaries) {
  const merged = {
    allowed_events: 0,
    object_events: 0,
    no_object_events: 0,
    camera_object_events: 0,
    ema_object_events: 0,
    suppression_verification_checks: 0,
    suppression_verification_rescues: 0,
    visual_backup_attempts: 0,
    visual_backup_objects: 0,
    visual_backup_no_object: 0,
    visual_backup_incomplete: 0,
  };
  for (const summary of summaries || []) {
    if (!summary || typeof summary !== "object") continue;
    for (const key of Object.keys(merged)) {
      merged[key] += asCount(summary[key]);
    }
  }
  if (!merged.camera_object_events && !merged.ema_object_events && merged.object_events) {
    merged.ema_object_events = asCount(summaryObjectEventsFromBackup(summaries));
    merged.camera_object_events = Math.max(0, merged.object_events - merged.ema_object_events);
  }
  return merged;
}

function summaryObjectEventsFromBackup(summaries) {
  return (summaries || []).reduce((total, summary) => (
    total + Math.min(asCount(summary?.visual_backup_objects), asCount(summary?.object_events))
  ), 0);
}

export function cameraEffectiveness(byCamera, cameraId, mode) {
  const modes = byCamera?.[cameraId] || {};
  if (mode && modes[mode]) return modes[mode];
  return mergeEffectiveness(Object.values(modes));
}

export function siteEffectiveness(byCamera, cameras = []) {
  const knownIds = new Set((cameras || []).map((camera) => camera.id).filter(Boolean));
  const summaries = [];
  Object.entries(byCamera || {}).forEach(([cameraId, modes]) => {
    if (knownIds.size && !knownIds.has(cameraId)) return;
    summaries.push(mergeEffectiveness(Object.values(modes || {})));
  });
  return mergeEffectiveness(summaries);
}

export function coverageFromRuntimeHistory(history) {
  const totals = (history || []).reduce((total, point) => ({
    analyzed: total.analyzed + asCount(point?.analysis_frames_sampled),
    superseded: total.superseded + asCount(point?.analysis_frames_dropped),
  }), { analyzed: 0, superseded: 0 });
  const sampled = totals.analyzed + totals.superseded;
  return {
    analyzed: totals.analyzed,
    staleSkipped: totals.superseded,
    coveragePercent: sampled ? (totals.analyzed / sampled) * 100 : null,
  };
}

export function coverageFromCameraMotion(camera) {
  const analysis = camera?.motion?.analysis_runtime || {};
  const analyzed = asCount(analysis.frames_sampled);
  const staleSkipped = asCount(analysis.mailbox_replacements || camera?.motion?.analysis_frames_dropped);
  const sampled = analyzed + staleSkipped;
  return {
    analyzed,
    staleSkipped,
    deferred: asCount(analysis.analysis_slot_deferrals),
    captureToAnalysisP95Ms: Number(analysis.capture_to_analysis_p95_ms || 0) || 0,
    coveragePercent: sampled ? (analyzed / sampled) * 100 : null,
  };
}

export function emaCoverageVerdict({
  coveragePercent,
  deferred = 0,
  staleSkipped = 0,
  slotCount = 2,
} = {}) {
  const slots = Math.max(1, asCount(slotCount) || 2);
  const nextSlots = Math.min(16, slots + 1);
  const setting = {
    workspace: "general",
    detectionSection: "motion",
    label: "Simultaneous EMA cameras",
  };
  if (coveragePercent == null) {
    return {
      id: "coverage",
      title: "Visual analysis capacity",
      tone: OCCUPANCY_TONES.idle,
      headline: "No recent visual samples",
      detail: "Visual backup is not sampling right now, so this limit is not under load.",
      suggestion: "Leave Simultaneous EMA cameras as it is.",
      setting,
    };
  }
  const coverage = `${Number(coveragePercent).toFixed(coveragePercent >= 99.95 ? 2 : 1)}%`;
  const waitNote = deferred
    ? `${asCount(deferred).toLocaleString()} times a camera waited for a free analysis slot.`
    : "Cameras are not waiting for a free analysis slot.";
  const staleNote = staleSkipped
    ? ` ${asCount(staleSkipped).toLocaleString()} older frames were replaced so analysis stayed current.`
    : "";
  if (coveragePercent >= 95 && !deferred) {
    return {
      id: "coverage",
      title: "Visual analysis capacity",
      tone: OCCUPANCY_TONES.good,
      headline: `${coverage} of visual samples were analyzed`,
      detail: `${waitNote}${staleNote}`,
      suggestion: `Leave Simultaneous EMA cameras at ${slots}.`,
      setting,
    };
  }
  if (coveragePercent >= 98) {
    return {
      id: "coverage",
      title: "Visual analysis capacity",
      tone: OCCUPANCY_TONES.good,
      headline: `${coverage} of visual samples were analyzed`,
      detail: `${waitNote}${staleNote}`,
      suggestion: `Leave Simultaneous EMA cameras at ${slots}.`,
      setting,
    };
  }
  if (deferred && coveragePercent < 80) {
    return {
      id: "coverage",
      title: "Visual analysis capacity",
      tone: OCCUPANCY_TONES.bad,
      headline: `Visual analysis is falling behind (${coverage})`,
      detail: `${waitNote}${staleNote}`,
      suggestion: `Raise Simultaneous EMA cameras from ${slots} to ${nextSlots}, then save and watch this card.`,
      setting,
    };
  }
  if (deferred && coveragePercent < 95) {
    return {
      id: "coverage",
      title: "Visual analysis capacity",
      tone: OCCUPANCY_TONES.warning,
      headline: `Some cameras are waiting in line (${coverage} coverage)`,
      detail: `${waitNote}${staleNote}`,
      suggestion: `Raise Simultaneous EMA cameras from ${slots} to ${nextSlots}.`,
      setting,
    };
  }
  return {
    id: "coverage",
    title: "Visual analysis capacity",
    tone: OCCUPANCY_TONES.warning,
    headline: `Coverage dipped to ${coverage}`,
    detail: `${waitNote}${staleNote} That is usually a camera stream issue, not this limit.`,
    suggestion: "Do not raise Simultaneous EMA cameras. Check camera video health first.",
    setting,
  };
}

export function doubleCheckVerdict({ checks = 0, rescues = 0 } = {}) {
  const setting = {
    workspace: "general",
    detectionSection: "motion",
    label: "Double-check filtered motion",
  };
  const checkCount = asCount(checks);
  const rescueCount = asCount(rescues);
  if (!checkCount) {
    return {
      id: "double-check",
      title: "Extra check on filtered motion",
      tone: OCCUPANCY_TONES.good,
      headline: "This extra detector check is idle",
      detail: "SurvNG is not spending detector time re-checking motion it already filtered out.",
      suggestion: "Leave Double-check filtered motion as it is.",
      setting,
    };
  }
  if (rescueCount) {
    return {
      id: "double-check",
      title: "Extra check on filtered motion",
      tone: OCCUPANCY_TONES.good,
      headline: `${rescueCount.toLocaleString()} filtered event${rescueCount === 1 ? "" : "s"} restored after finding an object`,
      detail: `${checkCount.toLocaleString()} extra detector check${checkCount === 1 ? "" : "s"} ran in the last 7 days.`,
      suggestion: "Leave Double-check filtered motion as it is. It is catching real objects.",
      setting,
    };
  }
  if (checkCount >= 50) {
    return {
      id: "double-check",
      title: "Extra check on filtered motion",
      tone: OCCUPANCY_TONES.bad,
      headline: `${checkCount.toLocaleString()} extra checks and no restored incidents`,
      detail: "The detector is being used on motion SurvNG already filtered, and nothing useful came back.",
      suggestion: "Set Double-check filtered motion to Off, or About 1 in 100.",
      setting,
    };
  }
  if (checkCount >= 20) {
    return {
      id: "double-check",
      title: "Extra check on filtered motion",
      tone: OCCUPANCY_TONES.warning,
      headline: `${checkCount.toLocaleString()} extra checks and no restored incidents`,
      detail: "This safety net is costing detector time without paying back yet.",
      suggestion: "Lower Double-check filtered motion to About 1 in 100.",
      setting,
    };
  }
  return {
    id: "double-check",
    title: "Extra check on filtered motion",
    tone: OCCUPANCY_TONES.idle,
    headline: `${checkCount.toLocaleString()} extra check${checkCount === 1 ? "" : "s"} and no restores yet`,
    detail: "That is too few samples to change the setting.",
    suggestion: "Leave Double-check filtered motion as it is for now.",
    setting,
  };
}

export function incidentSplitVerdict({
  cameraObjects = 0,
  emaObjects = 0,
  backupEnabled = true,
  onvifHealthy = true,
} = {}) {
  const setting = backupEnabled
    ? { workspace: "general", detectionSection: "motion", label: "Motion behavior" }
    : { workspace: "general", detectionSection: "motion", label: "Motion behavior" };
  const cameraCount = asCount(cameraObjects);
  const emaCount = asCount(emaObjects);
  const total = cameraCount + emaCount;
  if (!total) {
    return {
      id: "incident-split",
      title: "What starts an incident",
      tone: OCCUPANCY_TONES.idle,
      headline: "No incidents in the last 7 days",
      detail: "There is not enough activity to judge camera notices versus visual backup.",
      suggestion: "Nothing to change.",
      setting,
      cameraCount,
      emaCount,
      emaShare: null,
    };
  }
  const emaShare = emaCount / total;
  const percent = Math.round(emaShare * 100);
  const headline = `${cameraCount.toLocaleString()} from camera notices · ${emaCount.toLocaleString()} from visual backup`;
  if (emaShare <= 0.35) {
    return {
      id: "incident-split",
      title: "What starts an incident",
      tone: OCCUPANCY_TONES.good,
      headline,
      detail: "Most incidents start from the camera's own motion notice.",
      suggestion: backupEnabled
        ? "Keep Camera + EMA backup. It is only filling gaps."
        : "Camera notices are carrying the load.",
      setting,
      cameraCount,
      emaCount,
      emaShare,
    };
  }
  if (!onvifHealthy) {
    return {
      id: "incident-split",
      title: "What starts an incident",
      tone: OCCUPANCY_TONES.warning,
      headline,
      detail: `${percent}% started from visual backup because camera events look unhealthy.`,
      suggestion: "Keep Camera + EMA backup. Fix the camera event connection first. Do not switch to camera notices only.",
      setting,
      cameraCount,
      emaCount,
      emaShare,
    };
  }
  if (!backupEnabled) {
    return {
      id: "incident-split",
      title: "What starts an incident",
      tone: OCCUPANCY_TONES.idle,
      headline,
      detail: "Backup is off now. Older visual-backup incidents can still appear in this 7-day count.",
      suggestion: "Nothing to change unless you still want backup for missed camera notices.",
      setting,
      cameraCount,
      emaCount,
      emaShare,
    };
  }
  return {
    id: "incident-split",
    title: "What starts an incident",
    tone: OCCUPANCY_TONES.warning,
    headline,
    detail: `${percent}% started from visual backup because the camera notice was late or missing. The first event in a group keeps that label even if a camera notice arrives later.`,
    suggestion: "Keep Camera + EMA backup. Do not switch to camera notices only. Check camera event connection on the cameras with the highest backup share.",
    setting,
    cameraCount,
    emaCount,
    emaShare,
  };
}

export function visualBackupVerdict({
  attempts = 0,
  objects = 0,
  none = 0,
  incomplete = 0,
} = {}) {
  const setting = { workspace: "audit", category: "visual_backup", label: "Motion Audit" };
  const attemptCount = asCount(attempts);
  const objectCount = asCount(objects);
  if (!attemptCount) {
    return {
      id: "visual-backup",
      title: "Visual backup results",
      tone: OCCUPANCY_TONES.idle,
      headline: "Backup has not needed to run",
      detail: "Camera notices have been enough, or backup has not seen strong persistent motion.",
      suggestion: "Nothing to change.",
      setting,
    };
  }
  if (objectCount) {
    return {
      id: "visual-backup",
      title: "Visual backup results",
      tone: OCCUPANCY_TONES.good,
      headline: `${objectCount.toLocaleString()} backup run${objectCount === 1 ? "" : "s"} found a real object`,
      detail: `${attemptCount.toLocaleString()} backup attempt${attemptCount === 1 ? "" : "s"} · ${asCount(none).toLocaleString()} found none${incomplete ? ` · ${asCount(incomplete).toLocaleString()} incomplete` : ""}.`,
      suggestion: "Keep Camera + EMA backup. It is catching missed camera notices.",
      setting,
    };
  }
  return {
    id: "visual-backup",
    title: "Visual backup results",
    tone: OCCUPANCY_TONES.warning,
    headline: `${attemptCount.toLocaleString()} backup run${attemptCount === 1 ? "" : "s"} and no objects found`,
    detail: "Backup used the detector without creating a useful incident.",
    suggestion: "Open Motion Audit, category Visual backup, and review those pictures before changing motion behavior.",
    setting,
  };
}

const PARALLEL_DETECTORS_SETTING = Object.freeze({
  workspace: "general",
  detectionSection: "object",
  label: "Parallel detectors",
});

const OBJECT_DETECTION_SETTING = Object.freeze({
  workspace: "general",
  detectionSection: "object",
  label: "Object Detection",
});

const TRACKING_SETTING = Object.freeze({
  workspace: "general",
  detectionSection: "tracking",
  label: "Tracking & ReID",
});

const HIGH_ADMISSION_WAIT_MS = 100;
const QUEUE_WARN_PER_WORKER = 1;
const FAILED_INFERENCE_WARN = 3;
const FAILED_INFERENCE_BAD = 15;

function optionalCount(value) {
  if (value == null || value === "") return null;
  const count = Number(value);
  return Number.isFinite(count) && count >= 0 ? Math.round(count) : null;
}

function optionalNumber(value) {
  if (value == null || value === "") return null;
  const count = Number(value);
  return Number.isFinite(count) ? count : null;
}

function detectorFinding(tone, headline, detail, suggestion, setting = PARALLEL_DETECTORS_SETTING) {
  return { tone, headline, detail, suggestion, setting };
}

export function resolveObjectWorkerCount({ config, telemetry } = {}) {
  const objectWorkers = telemetry?.detector?.workers?.object
    || telemetry?.detector?.isolation
    || {};
  const fromConfig = optionalCount(config?.detector?.object_worker_count);
  const spawned = optionalCount(
    objectWorkers.configured_workers
    ?? telemetry?.detector?.object_worker_count,
  );
  const alive = optionalCount(objectWorkers.alive_workers);
  const configured = fromConfig ?? spawned ?? 2;
  const running = alive ?? spawned ?? configured;
  return { configured, running, alive, spawned };
}

export function resolveDetectorHealth({ config, telemetry } = {}) {
  const detector = telemetry?.detector || {};
  const runtime = detector.runtime || {};
  const objectWorkers = detector.workers?.object || detector.isolation || {};
  const counts = resolveObjectWorkerCount({ config, telemetry });
  const workloads = runtime.workloads || {};
  const classes = workloads.classes || {};
  const initialWait = optionalNumber(classes.incident_initial?.admission_wait_ms_p95);
  const trackingCapacity = (telemetry?.tracking_capacity_history?.short || []).reduce(
    (total, point) => ({
      attempts: total.attempts + asCount(point?.attempts),
      waited: total.waited + asCount(point?.waited),
      skipped: total.skipped + asCount(point?.skipped),
    }),
    { attempts: 0, waited: 0, skipped: 0 },
  );
  const enabledSetting = config?.detector?.enabled !== false;
  const enabledRuntime = detector.enabled !== false;
  const hasLiveDetector = Boolean(
    detector.loaded_backend
    || detector.configured_backend
    || detector.isolation
    || detector.workers
    || detector.runtime
    || detector.enabled != null,
  );
  return {
    ...counts,
    enabled: enabledSetting,
    runtimeEnabled: enabledRuntime,
    trackingEnabled: config?.detector?.tracking?.enabled !== false,
    backend: String(config?.detector?.backend || detector.configured_backend || "openvino"),
    loaded: hasLiveDetector
      ? Boolean(detector.loaded_backend || detector.openvino_loaded || detector.coreml_loaded)
      : true,
    loadedBackend: String(detector.loaded_backend || ""),
    loadedDevice: String(detector.loaded_device || detector.configured_device || objectWorkers.configured_device || ""),
    fallbackActive: Boolean(objectWorkers.fallback_active),
    pending: optionalCount(objectWorkers.pending_requests) ?? optionalCount(runtime.queue_depth) ?? 0,
    failed: optionalCount(runtime.failed_inferences) ?? 0,
    crashes: optionalCount(objectWorkers.crash_count) ?? 0,
    restarts: optionalCount(objectWorkers.restart_count) ?? 0,
    lastError: String(objectWorkers.last_error || detector.warmup_error || "").trim(),
    initialWaitMs: initialWait,
    initialWaiting: optionalCount(workloads.initial_waiting) ?? 0,
    trackingSkipped: trackingCapacity.skipped,
    trackingAttempts: trackingCapacity.attempts,
    trackingWaited: trackingCapacity.waited,
  };
}

export function detectorLaneVerdict({
  trackingEnabled = false,
  workerCount = 0,
  configuredWorkerCount = null,
  runningWorkerCount = null,
  configured: configuredCount = null,
  running: runningCount = null,
  backend = "openvino",
  enabled = true,
  runtimeEnabled = true,
  loaded = true,
  loadedDevice = "",
  fallbackActive = false,
  pending = 0,
  failed = 0,
  crashes = 0,
  restarts = 0,
  lastError = "",
  initialWaitMs = null,
  initialWaiting = 0,
  trackingSkipped = 0,
  includeTracking = true,
} = {}) {
  const configured = optionalCount(configuredWorkerCount) ?? optionalCount(configuredCount) ?? optionalCount(workerCount) ?? 0;
  const running = optionalCount(runningWorkerCount) ?? optionalCount(runningCount) ?? configured;
  const workers = Math.max(1, running || configured);
  const pendingCount = optionalCount(pending) ?? 0;
  const failedCount = optionalCount(failed) ?? 0;
  const crashCount = optionalCount(crashes) ?? 0;
  const restartCount = optionalCount(restarts) ?? optionalCount(crashCount) ?? 0;
  const findings = [];

  if (enabled === false) {
    findings.push(detectorFinding(
      OCCUPANCY_TONES.idle,
      "Detector is turned off",
      "Object incidents will not start until the detector is enabled.",
      "Turn Detector enabled on if you want SurvNG to recognize objects.",
      OBJECT_DETECTION_SETTING,
    ));
  } else if (loaded === false || runtimeEnabled === false) {
    findings.push(detectorFinding(
      OCCUPANCY_TONES.bad,
      "Detector model is not loaded",
      "The detector process is on, but it does not have a working model.",
      "Open Object Detection, confirm the model path, then save and wait for it to load.",
      OBJECT_DETECTION_SETTING,
    ));
  } else if (running === 0) {
    findings.push(detectorFinding(
      OCCUPANCY_TONES.bad,
      "No detector process is running",
      configured
        ? `Parallel detectors is ${configured}, but none of those processes are up.`
        : "SurvNG has no live detector process to score objects.",
      "Open Object Detection, save if you just changed it, then wait for the detector processes to start.",
      OBJECT_DETECTION_SETTING,
    ));
  } else if (String(backend || "") !== "openvino") {
    findings.push(detectorFinding(
      OCCUPANCY_TONES.good,
      "This detector uses one worker",
      "Core ML keeps a single detector process.",
      "Nothing to change.",
      OBJECT_DETECTION_SETTING,
    ));
  } else {
    if (configured && running !== configured) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        `${running} detector${running === 1 ? "" : "s"} running · Parallel detectors is ${configured}`,
        "The saved setting and the detector processes that are up do not match yet.",
        "Open Parallel detectors to confirm the setting is saved, then wait for the detector processes to restart.",
      ));
    }
    if (includeTracking && trackingEnabled && workers < 2) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.bad,
        "Tracking is on with only one detector",
        "Overlay work can occupy the only detector while a new incident needs a yes or no.",
        "Set Parallel detectors to 2, or turn tracking off if you only need snapshots.",
      ));
    }
  }

  if (enabled !== false && running > 0) {
    if (fallbackActive) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        "Detector is on CPU fallback",
        "The accelerator failed, so object scoring is running on CPU until it recovers.",
        "Check Object Detection device. If this stays on CPU, GPU or NPU is not available.",
        OBJECT_DETECTION_SETTING,
      ));
    }
    const queuePressure = Math.max(1, (configured || workers) * QUEUE_WARN_PER_WORKER);
    if (pendingCount >= queuePressure * 2 || (optionalCount(initialWaiting) || 0) >= 2) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.bad,
        `${pendingCount.toLocaleString()} detection${pendingCount === 1 ? "" : "s"} waiting in queue`,
        "New incident checks are backing up behind work already on the detector.",
        configured >= 4
          ? "The detector is at its worker limit. Check camera load and tracking before changing anything else."
          : "Raise Parallel detectors by 1, then watch this queue.",
      ));
    } else if (pendingCount >= queuePressure) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        `${pendingCount.toLocaleString()} detection${pendingCount === 1 ? "" : "s"} waiting in queue`,
        "The detector has a short backlog. That is only a problem if it stays here.",
        configured >= 4
          ? "Leave Parallel detectors at 4 and watch Telemetry detector response."
          : "If this stays queued, raise Parallel detectors by 1.",
      ));
    }
    if (initialWaitMs != null && initialWaitMs >= HIGH_ADMISSION_WAIT_MS) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        `Live incident checks are waiting ${Math.round(initialWaitMs)} ms`,
        "Admission wait is high enough that a new object can miss its window.",
        configured >= 4
          ? "Workers are already at the limit. Check tracking load and camera count."
          : "Raise Parallel detectors by 1 if this wait stays high.",
      ));
    }
    if (failedCount >= FAILED_INFERENCE_BAD) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.bad,
        `${failedCount.toLocaleString()} detector failures since restart`,
        "The model is erroring often enough to miss objects.",
        "Open Object Detection and confirm the model and device, then watch Detector response on Telemetry.",
        OBJECT_DETECTION_SETTING,
      ));
    } else if (failedCount >= FAILED_INFERENCE_WARN) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        `${failedCount.toLocaleString()} detector failures since restart`,
        "A few failed inferences are normal after a restart. A rising count is not.",
        "Watch Detector response on Telemetry. If failures keep climbing, check the model and device.",
        OBJECT_DETECTION_SETTING,
      ));
    }
    if (crashCount >= 1) {
      findings.push(detectorFinding(
        running > 0 && running === configured ? OCCUPANCY_TONES.warning : OCCUPANCY_TONES.bad,
        `Detector restarted after ${crashCount.toLocaleString()} crash${crashCount === 1 ? "" : "es"}`,
        lastError || "The process came back. Repeated crashes mean the model or device needs a look.",
        "If crashes continue, open Object Detection and confirm the model and accelerator.",
        OBJECT_DETECTION_SETTING,
      ));
    } else if (lastError) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        "Detector reported an error",
        lastError,
        "Open Object Detection and confirm the model and device if this error stays.",
        OBJECT_DETECTION_SETTING,
      ));
    }
    if (includeTracking && trackingEnabled && trackingSkipped >= 1 && workers >= 2) {
      findings.push(detectorFinding(
        OCCUPANCY_TONES.warning,
        `${trackingSkipped.toLocaleString()} tracking session${trackingSkipped === 1 ? "" : "s"} skipped in 2 hours`,
        "Tracking could not get a detector lane, so some overlays were dropped.",
        configured >= 4
          ? "Workers are at the limit. Reduce simultaneous tracking or leave overlays off on busy cameras."
          : "Raise Parallel detectors by 1 if skips keep happening.",
      ));
    }
  }

  const actionable = findings.filter((item) => item.tone !== OCCUPANCY_TONES.good);
  const primary = actionable.sort((left, right) => (
    (TONE_RANK[right.tone] || 0) - (TONE_RANK[left.tone] || 0)
  ))[0];
  if (primary) {
    const extras = actionable.filter((item) => item !== primary).map((item) => item.headline);
    return {
      id: "detector-lane",
      title: "Detection engine",
      tone: primary.tone,
      headline: primary.headline,
      detail: [primary.detail, extras.length ? extras.join(" · ") : ""].filter(Boolean).join(" "),
      suggestion: primary.suggestion,
      setting: primary.setting,
    };
  }

  const device = String(loadedDevice || "").trim();
  const extras = [
    pendingCount ? `${pendingCount.toLocaleString()} waiting` : "no queue",
    failedCount ? `${failedCount.toLocaleString()} failures since restart` : "no failures",
    includeTracking ? (trackingEnabled ? "tracking keeps a reserved lane" : "tracking is off") : "",
    device ? device : "",
  ].filter(Boolean);
  if (String(backend || "") !== "openvino") {
    return {
      id: "detector-lane",
      title: "Detection engine",
      tone: OCCUPANCY_TONES.good,
      headline: "This detector uses one worker",
      detail: extras.join(" · "),
      suggestion: "Nothing to change.",
      setting: OBJECT_DETECTION_SETTING,
    };
  }
  return {
    id: "detector-lane",
    title: "Detection engine",
    tone: OCCUPANCY_TONES.good,
    headline: includeTracking && trackingEnabled
      ? `${workers} detectors · tracking keeps a reserved lane`
      : `${workers} detector${workers === 1 ? "" : "s"} running`,
    detail: extras.join(" · "),
    suggestion: workers === 1 && !trackingEnabled
      ? "One detector is enough while tracking is off."
      : "Leave Parallel detectors as it is.",
    setting: PARALLEL_DETECTORS_SETTING,
  };
}

export function trackingHealthVerdict({
  trackingEnabled = false,
  workerCount = 0,
  configuredWorkerCount = null,
  runningWorkerCount = null,
  configured: configuredCount = null,
  running: runningCount = null,
  backend = "openvino",
  trackingSkipped = 0,
  trackingAttempts = 0,
  trackingWaited = 0,
} = {}) {
  const configured = optionalCount(configuredWorkerCount) ?? optionalCount(configuredCount) ?? optionalCount(workerCount) ?? 0;
  const running = optionalCount(runningWorkerCount) ?? optionalCount(runningCount) ?? configured;
  const workers = Math.max(1, running || configured);
  const skipped = asCount(trackingSkipped);
  const attempts = asCount(trackingAttempts);
  const waited = asCount(trackingWaited);
  if (!trackingEnabled) {
    return {
      id: "tracking",
      title: "Tracking",
      tone: OCCUPANCY_TONES.idle,
      headline: "Tracking is off",
      detail: "Overlays and identity work are not using a detector lane.",
      suggestion: "Nothing to change unless you want live overlays.",
      setting: TRACKING_SETTING,
    };
  }
  if (String(backend || "") === "openvino" && workers < 2) {
    return {
      id: "tracking",
      title: "Tracking",
      tone: OCCUPANCY_TONES.bad,
      headline: "Tracking is on with only one detector",
      detail: "Overlay work can occupy the only detector while a new incident needs a yes or no.",
      suggestion: "Set Parallel detectors to 2, or turn tracking off if you only need snapshots.",
      setting: PARALLEL_DETECTORS_SETTING,
    };
  }
  if (skipped >= 1) {
    return {
      id: "tracking",
      title: "Tracking",
      tone: OCCUPANCY_TONES.warning,
      headline: `${skipped.toLocaleString()} tracking session${skipped === 1 ? "" : "s"} skipped in 2 hours`,
      detail: attempts
        ? `${attempts.toLocaleString()} tracking session${attempts === 1 ? "" : "s"} · ${waited.toLocaleString()} waited for a lane.`
        : "Tracking could not get a detector lane, so some overlays were dropped.",
      suggestion: configured >= 4
        ? "Workers are at the limit. Reduce simultaneous tracking or leave overlays off on busy cameras."
        : "Raise Parallel detectors by 1 if skips keep happening.",
      setting: PARALLEL_DETECTORS_SETTING,
    };
  }
  const extras = [
    attempts ? `${attempts.toLocaleString()} session${attempts === 1 ? "" : "s"} in 2 hours` : "no skipped sessions",
    waited ? `${waited.toLocaleString()} waited briefly` : "no waits",
    String(backend || "") === "openvino" ? "reserved detector lane" : "single-worker detector",
  ];
  return {
    id: "tracking",
    title: "Tracking",
    tone: OCCUPANCY_TONES.good,
    headline: "Tracking has a reserved detector lane",
    detail: extras.join(" · "),
    suggestion: "Leave tracking as it is.",
    setting: TRACKING_SETTING,
  };
}

export function incidentEligibilityRow({ requireZone = true } = {}) {
  return {
    id: "eligibility",
    title: "What can become an incident",
    tone: OCCUPANCY_TONES.idle,
    headline: requireZone ? "Only objects inside a zone" : "Zones plus the whole frame",
    detail: requireZone
      ? "Ignore zones still hide their object classes either way."
      : "A detection anywhere in the picture can become an incident unless an ignore zone hides that class.",
    suggestion: "Change Incident eligibility only if this does not match how you use zones.",
    setting: {
      workspace: "general",
      detectionSection: "object",
      label: "Incident eligibility",
    },
  };
}

export function occupancyToneLabel(tone) {
  if (tone === OCCUPANCY_TONES.bad) return "Needs attention";
  if (tone === OCCUPANCY_TONES.warning) return "Have a look";
  if (tone === OCCUPANCY_TONES.good) return "Looks good";
  return "No action";
}

const PILLAR_VOICE = Object.freeze({
  admission: "admission",
  engine: "worker capacity",
  "detector-lane": "worker capacity",
  tracking: "tracking",
  capacity: "visual analysis",
  coverage: "visual analysis",
});

const SUMMARY_PILLAR_ORDER = Object.freeze(["admission", "tracking", "capacity", "engine"]);

function joinHealthNames(names) {
  if (!names.length) return "Detection";
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
}

function mergeHealthPillar(id, title, rows) {
  const present = (rows || []).filter(Boolean);
  if (!present.length) return null;
  const tone = worstOccupancyTone(present.map((row) => row.tone));
  const ranked = [...present].sort((left, right) => (
    (TONE_RANK[right.tone] || 0) - (TONE_RANK[left.tone] || 0)
  ));
  const primary = ranked[0];
  const extras = ranked.slice(1);
  let headline = primary.headline;
  let detail = primary.detail;
  if (id === "admission") {
    const split = present.find((row) => row.id === "incident-split");
    const backup = present.find((row) => row.id === "visual-backup");
    if (split && backup) {
      if ((TONE_RANK[tone] || 0) <= TONE_RANK[OCCUPANCY_TONES.good]) {
        headline = split.headline;
        detail = [split.detail, backup.headline].filter(Boolean).join(" ");
      } else {
        detail = [primary.detail, extras.map((row) => row.headline).filter(Boolean).join(" · ")].filter(Boolean).join(" ");
      }
    }
  } else if (extras.length) {
    detail = [primary.detail, extras.map((row) => row.headline).filter(Boolean).join(" · ")].filter(Boolean).join(" ");
  }
  return {
    id,
    title,
    tone,
    headline,
    detail,
    suggestion: primary.suggestion,
    setting: primary.setting,
    findings: present,
  };
}

export function groupDetectionHealth(rows = []) {
  const byId = Object.fromEntries((rows || []).filter((row) => row?.id).map((row) => [row.id, row]));
  const waste = byId["double-check"];
  return [
    mergeHealthPillar("admission", "Admission", [byId["incident-split"], byId["visual-backup"]]),
    byId["detector-lane"] ? {
      ...byId["detector-lane"],
      id: "engine",
      title: "Detection engine",
      findings: [byId["detector-lane"]],
    } : null,
    byId.tracking || null,
    byId.coverage ? {
      ...byId.coverage,
      id: "capacity",
      title: "Visual analysis",
      findings: [byId.coverage],
    } : null,
    waste && (waste.tone === OCCUPANCY_TONES.warning || waste.tone === OCCUPANCY_TONES.bad)
      ? { ...waste, id: "waste", title: "Extra detector checks", findings: [waste] }
      : null,
  ].filter(Boolean);
}

export function occupancyReportSummary(items = []) {
  const engine = items.find((item) => item.id === "engine" || item.id === "detector-lane");
  const attention = items.filter((item) => (
    item.tone === OCCUPANCY_TONES.warning || item.tone === OCCUPANCY_TONES.bad
  ));
  if (engine?.tone === OCCUPANCY_TONES.idle && !attention.length) {
    return {
      headline: engine.headline,
      detail: engine.detail,
    };
  }
  if (!attention.length) {
    const names = SUMMARY_PILLAR_ORDER
      .map((id) => items.find((item) => item.id === id))
      .filter((item) => item && item.tone !== OCCUPANCY_TONES.idle)
      .map((item) => PILLAR_VOICE[item.id] || String(item.title || "").toLowerCase())
      .filter(Boolean);
    return {
      headline: "Detection looks healthy",
      detail: names.length
        ? `${joinHealthNames(names)} ${names.length === 1 ? "is" : "are"} in a good range.`
        : "Live detection, visual backup, and extra checks are in a good range.",
    };
  }
  return {
    headline: worstOccupancyTone(attention.map((item) => item.tone)) === OCCUPANCY_TONES.bad
      ? "Detection needs attention"
      : "Detection needs a look",
    detail: attention.map((item) => item.headline).slice(0, 3).join(" · "),
  };
}

export function buildOccupancyReport({
  coverage,
  effectiveness,
  slotCount,
  trackingEnabled,
  workerCount,
  configuredWorkerCount,
  runningWorkerCount,
  backend,
  requireZone,
  backupEnabled,
  onvifHealthy,
  detectorHealth = null,
  includeDetectorHealth = true,
} = {}) {
  const trackingOn = detectorHealth?.trackingEnabled ?? trackingEnabled;
  const rows = [
    emaCoverageVerdict({
      coveragePercent: coverage?.coveragePercent,
      deferred: coverage?.deferred,
      staleSkipped: coverage?.staleSkipped,
      slotCount,
    }),
    incidentSplitVerdict({
      cameraObjects: effectiveness?.camera_object_events,
      emaObjects: effectiveness?.ema_object_events,
      backupEnabled,
      onvifHealthy,
    }),
    visualBackupVerdict({
      attempts: effectiveness?.visual_backup_attempts,
      objects: effectiveness?.visual_backup_objects,
      none: effectiveness?.visual_backup_no_object,
      incomplete: effectiveness?.visual_backup_incomplete,
    }),
    doubleCheckVerdict({
      checks: effectiveness?.suppression_verification_checks,
      rescues: effectiveness?.suppression_verification_rescues,
    }),
  ];
  if (includeDetectorHealth) {
    rows.push(detectorLaneVerdict({
      trackingEnabled,
      workerCount,
      configuredWorkerCount,
      runningWorkerCount,
      backend,
      includeTracking: false,
      ...(detectorHealth ? {
        trackingEnabled: detectorHealth.trackingEnabled,
        workerCount: detectorHealth.running,
        configuredWorkerCount: detectorHealth.configured,
        runningWorkerCount: detectorHealth.running,
        backend: detectorHealth.backend,
        enabled: detectorHealth.enabled,
        runtimeEnabled: detectorHealth.runtimeEnabled,
        loaded: detectorHealth.loaded,
        loadedDevice: detectorHealth.loadedDevice,
        fallbackActive: detectorHealth.fallbackActive,
        pending: detectorHealth.pending,
        failed: detectorHealth.failed,
        crashes: detectorHealth.crashes,
        restarts: detectorHealth.restarts,
        lastError: detectorHealth.lastError,
        initialWaitMs: detectorHealth.initialWaitMs,
        initialWaiting: detectorHealth.initialWaiting,
        trackingSkipped: detectorHealth.trackingSkipped,
      } : {}),
    }));
  }
  const trackingRow = includeDetectorHealth && trackingOn
    ? trackingHealthVerdict({
      trackingEnabled: trackingOn,
      workerCount,
      configuredWorkerCount,
      runningWorkerCount,
      backend,
      ...(detectorHealth ? {
        workerCount: detectorHealth.running,
        configuredWorkerCount: detectorHealth.configured,
        runningWorkerCount: detectorHealth.running,
        backend: detectorHealth.backend,
        trackingSkipped: detectorHealth.trackingSkipped,
        trackingAttempts: detectorHealth.trackingAttempts,
        trackingWaited: detectorHealth.trackingWaited,
      } : {}),
    })
    : null;
  const eligibility = incidentEligibilityRow({ requireZone });
  rows.push(eligibility);
  const pillars = groupDetectionHealth([
    ...rows,
    trackingRow,
  ].filter(Boolean));
  return {
    tone: worstOccupancyTone(pillars.map((pillar) => pillar.tone)),
    rows,
    pillars,
    context: eligibility.headline,
    summary: occupancyReportSummary(pillars),
  };
}
