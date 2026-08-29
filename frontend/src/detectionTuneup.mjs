export const TUNEUP_PERIODS = {
  quick: { label: "Last 24 hours", detail: "Fastest review · recent evidence · lowest AI usage" },
  standard: { label: "Last 7 days", detail: "Recommended · balances scene variety, speed, and AI usage" },
  deep: { label: "Last 30 days", detail: "Deepest evidence · slowest review · highest AI usage" },
};

export const TUNEUP_SETTING_NAMES = {
  "motion.sensitivity": "Motion sensitivity",
  "motion.stationary_object_tolerance": "Stationary object tolerance",
  "motion.visual_backup_min_score": "EMA rescue confidence",
  "motion.visual_backup_score_margin": "EMA rescue score margin",
  "motion.visual_backup_min_consecutive": "Motion persistence",
  "motion.visual_backup_cooldown_seconds": "Pause between EMA rescues",
  "motion.visual_backup_max_triggers_5m": "EMA rescue frequency",
  "motion.frame_width": "Motion analysis resolution",
  "detector.tracking.sample_fps": "Tracking frame rate",
};

export function tuneupValue(value) {
  if (value === null || value === undefined || value === "inherit") return "Use system default";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (typeof value === "string") return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  return String(value);
}

export function tuneupRecommendationGroup(item) {
  const setting = String(item.setting || "");
  const current = Number(item.current_effective ?? item.current);
  const proposed = Number(item.proposed);
  if (item.subsystem === "tracking") return "Improve identification overlays";
  if (["motion.frame_width", "motion.sample_fps"].includes(setting) && Number.isFinite(current) && proposed < current) return "Reduce processing load";
  if (["motion.visual_backup_min_score", "motion.visual_backup_min_consecutive", "motion.visual_backup_cooldown_seconds"].includes(setting) && Number.isFinite(current)) return proposed > current ? "Reduce unwanted motion" : "Catch more important activity";
  if (setting === "motion.sensitivity") return String(item.proposed) === "high" ? "Catch more important activity" : "Reduce unwanted motion";
  if (setting === "motion.stationary_object_tolerance") return "Reduce unwanted motion";
  return "Catch more important activity";
}

export function tuneupOutcome(item) {
  const outcome = String(item?.evaluation?.outcome || "inconclusive");
  if (outcome === "improved") return ["Improved", "good"];
  if (outcome === "mixed") return ["Mixed results", "warn"];
  if (outcome === "regressed") return ["Performance declined", "bad"];
  return ["No clear difference", "neutral"];
}

export function tuneupHistoryTitle(run, cameras) {
  const period = TUNEUP_PERIODS[run.mode]?.label?.replace("Last ", "") || "Custom review";
  const cameraIds = run.camera_ids || [];
  const cameraLabel = cameraIds.length === cameras.length ? "All cameras" : cameraIds.length === 1 ? (cameras.find((camera) => camera.id === cameraIds[0])?.name || cameraIds[0]) : `${cameraIds.length} cameras`;
  return `${period} review · ${cameraLabel}`;
}
