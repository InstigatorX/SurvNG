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

export function detectorLaneVerdict({
  trackingEnabled = false,
  workerCount = 2,
  backend = "openvino",
} = {}) {
  const setting = {
    workspace: "general",
    detectionSection: trackingEnabled ? "tracking" : "object",
    label: trackingEnabled ? "Parallel detectors / tracking" : "Parallel detectors",
  };
  if (String(backend || "") !== "openvino") {
    return {
      id: "detector-lane",
      title: "Detector workers",
      tone: OCCUPANCY_TONES.good,
      headline: "This detector uses one worker",
      detail: "Core ML keeps a single detector process.",
      suggestion: "Nothing to change.",
      setting: { ...setting, detectionSection: "object" },
    };
  }
  const workers = Math.max(1, asCount(workerCount) || 1);
  if (trackingEnabled && workers < 2) {
    return {
      id: "detector-lane",
      title: "Detector workers",
      tone: OCCUPANCY_TONES.bad,
      headline: "Tracking is on with only one detector",
      detail: "Overlay work can occupy the only detector while a new incident needs a yes or no.",
      suggestion: "Set Parallel detectors to 2, or turn tracking off if you only need snapshots.",
      setting,
    };
  }
  if (trackingEnabled) {
    return {
      id: "detector-lane",
      title: "Detector workers",
      tone: OCCUPANCY_TONES.good,
      headline: `${workers} detectors · tracking keeps a reserved lane`,
      detail: "New incident checks stay off the overlay queue.",
      suggestion: "Leave Parallel detectors as it is.",
      setting,
    };
  }
  return {
    id: "detector-lane",
    title: "Detector workers",
    tone: OCCUPANCY_TONES.good,
    headline: trackingEnabled ? `${workers} detectors` : `${workers} detector${workers === 1 ? "" : "s"} · tracking is off`,
    detail: "Incident checks are the only work on this queue.",
    suggestion: workers === 1
      ? "One detector is enough while tracking is off."
      : "Leave Parallel detectors as it is.",
    setting,
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

export function buildOccupancyReport({
  coverage,
  effectiveness,
  slotCount,
  trackingEnabled,
  workerCount,
  backend,
  requireZone,
  backupEnabled,
  onvifHealthy,
} = {}) {
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
    detectorLaneVerdict({ trackingEnabled, workerCount, backend }),
    incidentEligibilityRow({ requireZone }),
  ];
  return {
    tone: worstOccupancyTone(rows.map((row) => row.tone)),
    rows,
  };
}
