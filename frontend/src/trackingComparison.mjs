export const TRACKING_ENGINE_LABELS = {
  survng_hybrid: "Hybrid (current)",
  survng_hybrid_candidate: "Hybrid (candidate)",
  ultralytics_tracktrack: "TrackTrack",
  ultralytics_botsort: "BoT-SORT",
  ultralytics_fasttrack: "FastTrack (legacy)",
  ultralytics_deepocsort: "Deep OC-SORT (historic)",
};

export const TRACKING_SAMPLING_PROFILES = [
  { value: "fixed_2fps", label: "2 FPS" },
  { value: "fixed_075fps", label: "0.75 FPS" },
  { value: "sparse_gaps", label: "Sparse gaps" },
];

export function trackingEngineLabel(implementation) {
  return TRACKING_ENGINE_LABELS[implementation] || String(implementation || "Tracker").replaceAll("_", " ");
}

export function trackingComparisonEngines(comparison) {
  return Object.entries(comparison?.engines || {});
}

export function successfulTrackingComparisonEngines(comparison) {
  return trackingComparisonEngines(comparison).filter(([, engine]) => !engine?.error);
}

export function trackingComparisonRequestUrl(eventId, samplingProfile) {
  return `/api/events/${eventId}/tracking-comparison?duration_seconds=30&sampling_profile=${encodeURIComponent(samplingProfile)}`;
}

export function trackingComparisonResultsArtifact(comparison) {
  if (!comparison) return null;
  const { replay, ...result } = comparison;
  return result;
}

export function trackingVerdictLabel(verdict) {
  return verdict === "inconclusive" ? "No clear winner" : trackingEngineLabel(verdict);
}

export function trackingHistoryVerdictLabel(verdict, result = null) {
  if ((verdict === "ultralytics_botsort" || verdict === "ultralytics_deepocsort") && !result?.replay_id) {
    return `${trackingEngineLabel(verdict)} (historic)`;
  }
  return trackingVerdictLabel(verdict);
}

export function trackingHistorySummaryEntries(verdicts, items = []) {
  return Object.entries(verdicts || {})
    .filter(([verdict, count]) => verdict !== "unreviewed" && Number(count) > 0)
    .map(([verdict, count]) => {
      const matchingItems = items.filter((item) => item?.verdict === verdict);
      const currentResult = matchingItems.find((item) => item?.result?.replay_id)?.result || matchingItems[0]?.result;
      const label = trackingHistoryVerdictLabel(verdict, currentResult);
      return { verdict, count, label: verdict === "inconclusive" ? label : `${label} better` };
    });
}
