export const VERDICT_LABELS = Object.freeze({
  likely_miss: "Likely missed",
  likely_false_alarm: "Likely false alarm",
  likely_misclassification: "Wrong label",
  consistent: "Looked right",
  uncertain: "Not sure",
});

export const PHASE_LABELS = Object.freeze({
  watching: "Watching",
  checking: "Checking the result",
  result: "Result in",
  rolled_back: "Rolled back",
});

export const AUTO_STATUS_LABELS = Object.freeze({
  automatic: "Applies automatically on the next weekly run",
  approval: "Needs your approval",
  disabled: "Applying is turned off in Server settings",
  manual: "This one stays manual",
});

export const ORIGIN_LABELS = Object.freeze({
  automatic: "SurvNG applied this",
  you: "You applied this",
  rollback: "This put settings back",
});

export const OUTCOME_LABELS = Object.freeze({
  improved: "Improved",
  worsened: "Got worse",
  regressed: "Got worse",
  mixed: "Mixed",
  inconclusive: "No clear difference",
  failed: "The check failed",
});

const NOTABLE_VERDICTS = new Set(["likely_miss", "likely_false_alarm", "likely_misclassification"]);

export function formatReviewValue(value, unit = "") {
  if (value === null || value === undefined || value === "inherit") return "Site setting";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (unit === "percent" && typeof value === "number") return `${Math.round(value * 100)}%`;
  if (unit === "seconds" && typeof value === "number") {
    const rounded = Number.isInteger(value) ? String(value) : String(Math.round(value * 10) / 10);
    return `${rounded} ${rounded === "1" ? "second" : "seconds"}`;
  }
  if (unit === "fps" && typeof value === "number") return `${value} per second`;
  if (unit === "px" && typeof value === "number") return `${value} px wide`;
  if (unit === "choice" && typeof value === "string") {
    return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }
  if (typeof value === "string") return value.replaceAll("_", " ");
  return String(value);
}

export function verdictLabel(verdict) {
  return VERDICT_LABELS[verdict] || "Noted";
}

export function phaseLabel(phase) {
  return PHASE_LABELS[phase] || "Recorded";
}

export function outcomeLabel(outcome) {
  return OUTCOME_LABELS[outcome] || "No clear difference";
}

export function autoStatusLabel(status) {
  return AUTO_STATUS_LABELS[status] || "";
}

export function originLabel(origin) {
  return ORIGIN_LABELS[origin] || "Change recorded";
}

export function glanceSentence(glance) {
  if (!glance) return "No review yet.";
  const cameras = Number(glance.cameras) || 0;
  const analyzed = Number(glance.analyzed) || 0;
  if (!cameras && !analyzed) return glance.summary || "No review yet.";
  const suggestions = Number(glance.suggestions) || 0;
  const pictures = `${analyzed} ${analyzed === 1 ? "picture" : "pictures"}`;
  const cameraLabel = `${cameras} ${cameras === 1 ? "camera" : "cameras"}`;
  const suggestionLabel = `${suggestions} ${suggestions === 1 ? "suggestion" : "suggestions"}`;
  const failed = Number(glance.failed) || 0;
  const skipped = failed ? ` ${failed} ${failed === 1 ? "camera" : "cameras"} skipped.` : "";
  return `Looked at ${pictures} across ${cameraLabel}. ${suggestionLabel}.${skipped}`;
}

export function plainNote(text, guide = {}) {
  let next = String(text || "");
  const keys = Object.keys(guide).sort((left, right) => right.length - left.length);
  for (const key of keys) {
    next = next.replaceAll(key, guide[key]?.label || key);
  }
  return next;
}

export function effectText(item, guide = {}) {
  const current = item?.current_effective ?? item?.current;
  const proposed = item?.proposed;
  if (proposed === null || proposed === undefined || proposed === "inherit") return "";
  if (typeof proposed === "number" && typeof current === "number") {
    if (proposed === current) return "";
    return proposed > current ? guide.higher || "" : guide.lower || "";
  }
  if (typeof proposed === "boolean") return proposed ? guide.higher || "" : guide.lower || "";
  const order = { low: 0, balanced: 1, high: 2 };
  if (Object.hasOwn(order, proposed) && Object.hasOwn(order, current) && order[proposed] !== order[current]) {
    return order[proposed] > order[current] ? guide.higher || "" : guide.lower || "";
  }
  return "";
}

export function findingsByCamera(result) {
  const reports = result?.camera_reports || {};
  const summaries = result?.camera_summaries || [];
  const names = new Map(summaries.map((item) => [item.camera_id, item.camera_name || item.camera_id]));
  const ids = summaries.length ? summaries.map((item) => item.camera_id) : Object.keys(reports);
  return ids.map((cameraId) => {
    const samples = (reports[cameraId]?.samples || []).filter((sample) => sample?.image_url || sample?.summary);
    const notable = samples.filter((sample) => NOTABLE_VERDICTS.has(sample.verdict));
    return {
      cameraId,
      cameraName: names.get(cameraId) || cameraId,
      samples: (notable.length ? notable : samples).slice(0, 6),
      notable: notable.length,
    };
  }).filter((group) => group.samples.length);
}
