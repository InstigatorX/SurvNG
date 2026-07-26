const DEFAULT_SETTINGS = Object.freeze({
  policy: "audit",
  sources: ["mog2", "onvif"],
  sourceThresholds: { mog2: 0.5, onvif: 0.5 },
  sourceWeights: { primary: 1, mog2: 1, onvif: 1 },
  weightedThreshold: 0.5,
  activationFrames: 1,
  releaseFrames: 3,
  cooldownSeconds: 5,
  stateTimeoutSeconds: 10,
});

const GUIDED_IMPLEMENTATIONS = [
  "buffered_evidence_fusion",
  "score_event_state",
  "score_trigger",
];

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum, fallback) {
  return Math.min(maximum, Math.max(minimum, finiteNumber(value, fallback)));
}

export function defaultMotionDecisionSettings() {
  return structuredClone(DEFAULT_SETTINGS);
}

export function readMotionDecisionFusion(fusion) {
  if (fusion == null || (Array.isArray(fusion) && fusion.length === 0)) {
    return { custom: false, usesDefaults: true, settings: defaultMotionDecisionSettings() };
  }
  if (
    !Array.isArray(fusion)
    || fusion.length !== GUIDED_IMPLEMENTATIONS.length
    || fusion.some((stage, index) => stage?.implementation !== GUIDED_IMPLEMENTATIONS[index])
  ) {
    return { custom: true, usesDefaults: false, settings: defaultMotionDecisionSettings() };
  }

  const fusionOptions = fusion[0]?.options || {};
  const stateOptions = fusion[1]?.options || {};
  const defaults = defaultMotionDecisionSettings();
  return {
    custom: false,
    usesDefaults: false,
    settings: {
      policy: ["audit", "any", "all", "weighted"].includes(fusionOptions.policy)
        ? fusionOptions.policy
        : defaults.policy,
      sources: Array.isArray(fusionOptions.sources)
        ? fusionOptions.sources.filter((source) => source === "mog2" || source === "onvif")
        : defaults.sources,
      sourceThresholds: {
        mog2: clamp(fusionOptions.source_thresholds?.mog2, 0, 1, defaults.sourceThresholds.mog2),
        onvif: clamp(fusionOptions.source_thresholds?.onvif, 0, 1, defaults.sourceThresholds.onvif),
      },
      sourceWeights: {
        primary: clamp(fusionOptions.source_weights?.primary, 0, 10, defaults.sourceWeights.primary),
        mog2: clamp(fusionOptions.source_weights?.mog2, 0, 10, defaults.sourceWeights.mog2),
        onvif: clamp(fusionOptions.source_weights?.onvif, 0, 10, defaults.sourceWeights.onvif),
      },
      weightedThreshold: clamp(
        fusionOptions.weighted_threshold,
        0,
        1,
        defaults.weightedThreshold,
      ),
      activationFrames: Math.round(clamp(
        stateOptions.activation_frames,
        1,
        20,
        defaults.activationFrames,
      )),
      releaseFrames: Math.round(clamp(
        stateOptions.release_frames,
        1,
        20,
        defaults.releaseFrames,
      )),
      cooldownSeconds: clamp(
        stateOptions.cooldown_seconds,
        0,
        300,
        defaults.cooldownSeconds,
      ),
      stateTimeoutSeconds: clamp(
        stateOptions.state_timeout_seconds,
        0,
        300,
        defaults.stateTimeoutSeconds,
      ),
    },
  };
}

export function buildMotionDecisionFusion(settings) {
  const normalized = { ...defaultMotionDecisionSettings(), ...settings };
  const sources = [...new Set((normalized.sources || []).filter(
    (source) => source === "mog2" || source === "onvif",
  ))];
  return [
    {
      stage_id: "evidence_fusion",
      implementation: "buffered_evidence_fusion",
      options: {
        sources,
        policy: normalized.policy,
        source_thresholds: {
          mog2: clamp(normalized.sourceThresholds?.mog2, 0, 1, 0.5),
          onvif: clamp(normalized.sourceThresholds?.onvif, 0, 1, 0.5),
        },
        source_weights: {
          primary: clamp(normalized.sourceWeights?.primary, 0, 10, 1),
          mog2: clamp(normalized.sourceWeights?.mog2, 0, 10, 1),
          onvif: clamp(normalized.sourceWeights?.onvif, 0, 10, 1),
        },
        weighted_threshold: clamp(normalized.weightedThreshold, 0, 1, 0.5),
        minimum_sources: 1,
        require_warmed: true,
      },
    },
    {
      stage_id: "event_state",
      implementation: "score_event_state",
      options: {
        activation_frames: Math.round(clamp(normalized.activationFrames, 1, 20, 1)),
        release_frames: Math.round(clamp(normalized.releaseFrames, 1, 20, 3)),
        cooldown_seconds: clamp(normalized.cooldownSeconds, 0, 300, 5),
        state_timeout_seconds: clamp(normalized.stateTimeoutSeconds, 0, 300, 10),
      },
    },
    { stage_id: "trigger", implementation: "score_trigger", options: {} },
  ];
}
