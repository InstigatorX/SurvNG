// Health summary model: tiles, findings, camera reliability rows and
// pipeline trends from the /api/telemetry payload plus hourly availability.

import { uptimePercent, uptimeSlots } from "./adminOverview.mjs";

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function sum(values) {
  return values.reduce((total, value) => total + (finite(value) ? value : 0), 0);
}

/** Numeric values of one series key, dropping gaps. */
export function seriesValues(history = [], key) {
  return (Array.isArray(history) ? history : []).map((point) => point?.[key]).filter(finite);
}

/** Last numeric value in a series, or null. */
export function latestValue(history = [], key) {
  const values = seriesValues(history, key);
  return values.length ? values.at(-1) : null;
}

/**
 * Unavailable minutes implied by hourly availability slots. The current hour
 * is partial, so it only counts the minutes elapsed since it started.
 */
export function unavailableMinutes(slots = [], now = Date.now()) {
  return Math.round(sum((Array.isArray(slots) ? slots : []).map((slot) => {
    const value = slot?.availability_percent;
    if (!finite(value)) return 0;
    const started = new Date(slot.started_at).getTime();
    const elapsed = Number.isFinite(started) ? Math.min(60, Math.max(0, (now - started) / 60_000)) : 60;
    return ((100 - Math.min(100, Math.max(0, value))) / 100) * elapsed;
  })));
}

function cameraOfflineNow(camera) {
  return camera?.connected === false
    || camera?.capture_connectivity === false
    || ["offline", "disconnected", "error"].includes(camera?.capture_connectivity);
}

/** One row per camera in scope, in telemetry order. */
export function healthCameraRows(data, availability = {}, cameraId = "", now = Date.now()) {
  const cameras = Array.isArray(data?.cameras) ? data.cameras : [];
  return cameras.filter((camera) => !cameraId || camera.id === cameraId).map((camera) => {
    const slots = availability?.[camera.id] || [];
    return {
      id: camera.id,
      name: camera.name || camera.id,
      uptimePercent: uptimePercent(slots),
      unavailableMinutes: unavailableMinutes(slots, now),
      offlineNow: cameraOfflineNow(camera),
      reconnects: Number(camera.capture_reconnects || 0),
      interruptions: sum(slots.map((slot) => Number(slot?.capture_interruptions || 0))),
      events: Number(camera.activity?.last_24h?.events || 0),
      uptime: uptimeSlots(slots),
    };
  });
}

/**
 * Findings: only conditions outside the normal range, most severe first.
 * Action ids are resolved by the page: open-camera, diagnostics, logs, detection.
 */
export function healthFindings({ data, rows = [] } = {}) {
  const findings = [];
  // Availability problems first (worst first), then recovered interruptions.
  const byUptime = [...rows].sort((left, right) => (left.uptimePercent ?? 100) - (right.uptimePercent ?? 100));
  for (const row of byUptime) {
    const offlineNow = row.offlineNow;
    if (offlineNow || (row.uptimePercent !== null && row.uptimePercent < 99 && row.unavailableMinutes > 0)) {
      findings.push({
        id: `uptime-${row.id}`,
        severity: offlineNow || row.uptimePercent < 90 ? "critical" : "warning",
        title: offlineNow
          ? `${row.name} is offline now`
          : `${row.name} was unavailable for about ${row.unavailableMinutes} min in the last 24 h`,
        detail: row.uptimePercent === null ? "No availability history yet." : `${row.uptimePercent.toFixed(1)}% available over 24 h (about ${row.unavailableMinutes} min unavailable)${row.interruptions ? ` with ${row.interruptions} stream interruption${row.interruptions === 1 ? "" : "s"}` : ""}. Check the camera's network link and power if this repeats.`,
        actions: [
          { id: "diagnostics", label: "Capture diagnostics", cameraId: row.id },
          { id: "open-camera", label: "Open camera", cameraId: row.id },
        ],
      });
    } else if (row.interruptions >= 3) {
      findings.push({
        id: `interruptions-${row.id}`,
        severity: "warning",
        title: `${row.name}: ${row.interruptions} stream interruptions in the last 24 h`,
        detail: "SurvNG reconnected each time, so there is no gap in availability. Frequent interruptions usually mean an unstable network link or an overloaded camera.",
        actions: [{ id: "diagnostics", label: "Capture diagnostics", cameraId: row.id }],
      });
    }
  }
  const unexpected = Number(data?.interruption_summary?.unexpected || 0);
  if (unexpected > 0) {
    findings.push({
      id: "restarts",
      severity: "warning",
      title: `SurvNG restarted unexpectedly ${unexpected === 1 ? "once" : `${unexpected} times`} in the last 24 h`,
      detail: "Recording and detection stop while the service restarts.",
      actions: [{ id: "logs", label: "View logs" }],
    });
  }
  const failures = Number(data?.detector?.runtime?.failed_inferences || 0);
  if (failures > 0) {
    findings.push({
      id: "detector-failures",
      severity: "warning",
      title: `${failures.toLocaleString()} detector request${failures === 1 ? "" : "s"} failed since restart`,
      detail: "Failed requests are not checked for objects. Persistent failures usually point at the accelerator or model.",
      actions: [{ id: "detection", label: "Open Detection" }],
    });
  }
  const skipped = sum(seriesValues(data?.tracking_capacity_history?.short, "skipped"));
  if (skipped > 0) {
    findings.push({
      id: "tracking-skipped",
      severity: "warning",
      title: `Object tracking skipped ${skipped} session${skipped === 1 ? "" : "s"} in the last 2 h`,
      detail: "More cameras needed tracking at once than the configured capacity allows.",
      actions: [{ id: "detection", label: "Open Detection" }],
    });
  }
  const order = { critical: 0, warning: 1, info: 2 };
  return findings
    .map((item, index) => ({ item, index }))
    .sort((left, right) => (order[left.item.severity] - order[right.item.severity]) || left.index - right.index)
    .map(({ item }) => item);
}

/** Memory growth per hour across a history window, from first to last sample. */
export function memoryTrendBytesPerHour(history = [], key = "rss_bytes") {
  const points = (Array.isArray(history) ? history : [])
    .filter((point) => finite(point?.[key]) && point?.sampled_at)
    .map((point) => ({ at: new Date(point.sampled_at).getTime(), value: point[key] }))
    .filter((point) => Number.isFinite(point.at));
  if (points.length < 2) return null;
  const hours = (points.at(-1).at - points[0].at) / 3_600_000;
  if (hours < 1) return null;
  return (points.at(-1).value - points[0].value) / hours;
}

/**
 * How well object tracking kept up over tracking_capacity_history buckets.
 * A request is late when it waited for a free tracker slot or was skipped.
 * Skipped requests first wait up to the configured limit, so they are
 * already among the waits when that limit is above zero; when it is zero
 * nothing waits. The larger of the two counts is therefore a bucket's late
 * requests in either case.
 */
export function trackingCapacitySummary(history = []) {
  const points = Array.isArray(history) ? history : [];
  let requests = 0;
  let waited = 0;
  let skipped = 0;
  let late = 0;
  let longestWaitSeconds = 0;
  const lateSeries = points.map((point) => {
    const attempts = Math.max(0, Number(point?.attempts) || 0);
    const bucketWaited = Math.max(0, Number(point?.waited) || 0);
    const bucketSkipped = Math.max(0, Number(point?.skipped) || 0);
    const bucketLate = Math.min(attempts, Math.max(bucketWaited, bucketSkipped));
    requests += attempts;
    waited += bucketWaited;
    skipped += bucketSkipped;
    late += bucketLate;
    longestWaitSeconds = Math.max(longestWaitSeconds, Number(point?.wait_seconds_max) || 0);
    return bucketLate;
  });
  return {
    requests,
    waited,
    skipped,
    onTimePercent: requests ? ((requests - late) / requests) * 100 : null,
    longestWaitSeconds,
    lateSeries,
  };
}
