const DEFAULT_SETTINGS = Object.freeze({
  policy: "audit",
  sources: [],
  includePrimary: true,
  failOpen: true,
  sourceThresholds: { mog2: 0.5, onvif: 0.5 },
  sourceWeights: { primary: 1, mog2: 1, onvif: 1 },
  weightedThreshold: 0.5,
  activationFrames: 1,
  releaseFrames: 3,
  cooldownSeconds: 5,
  stateTimeoutSeconds: 10,
});

export const MOTION_MODE_OPTIONS = Object.freeze([
  Object.freeze({
    value: "camera",
    label: "Camera-triggered (Recommended)",
    status: "Camera ONVIF triggers",
    description: "Only camera ONVIF notices and manual tests can start object detection. Optional visual validators can confirm ordinary camera motion before detection runs.",
  }),
  Object.freeze({
    value: "camera_rescue",
    label: "Camera + visual backup",
    status: "Camera primary · SurvNG visual backup",
    description: "Camera ONVIF notices remain primary. Exceptionally strong, persistent visual motion can start object detection when the camera stays silent; an eligible object is still required for an incident.",
  }),
  Object.freeze({
    value: "adaptive",
    label: "Visual-triggered",
    status: "SurvNG adaptive triggers",
    description: "Adaptive visual motion starts object detection. Ordinary ONVIF notices are recorded as diagnostics but cannot trigger detection; MOG2 can optionally be required as confirmation.",
  }),
]);

const LEGACY_MODE_INFO = Object.freeze({
  audit: Object.freeze({
    value: "audit",
    label: "Legacy decision preview",
    status: "Legacy camera triggers · preview only",
    description: "Camera notices always run object detection; adaptive decisions are recorded but never enforced. Select a current mode to migrate this configuration.",
  }),
  off: Object.freeze({
    value: "off",
    label: "Legacy filtering off",
    status: "Legacy camera triggers · no filtering",
    description: "Camera notices always run object detection. Select Camera-triggered to migrate this configuration.",
  }),
  enforce: Object.freeze({
    value: "enforce",
    label: "Legacy hybrid mode",
    status: "Legacy camera + visual triggers",
    description: "Both ordinary ONVIF and adaptive visual motion may trigger detection. Select a current mode to remove this ambiguous hybrid behavior.",
  }),
});

export function motionModeInfo(mode) {
  if (LEGACY_MODE_INFO[mode]) return LEGACY_MODE_INFO[mode];
  return MOTION_MODE_OPTIONS.find((option) => option.value === mode)
    || MOTION_MODE_OPTIONS.find((option) => option.value === "camera");
}

const GUIDED_IMPLEMENTATIONS = [
  "buffered_evidence_fusion",
  "score_event_state",
  "score_trigger",
];
const GUIDED_STAGE_IDS = ["evidence_fusion", "event_state", "trigger"];
const GUIDED_OPTION_KEYS = [
  new Set([
    "sources",
    "policy",
    "source_thresholds",
    "source_weights",
    "weighted_threshold",
    "minimum_sources",
    "require_warmed",
    "include_primary",
    "fail_open",
  ]),
  new Set([
    "activation_frames",
    "release_frames",
    "cooldown_seconds",
    "state_timeout_seconds",
  ]),
  new Set(),
];

function isGuidedStage(stage, index) {
  const options = stage?.options || {};
  return stage?.implementation === GUIDED_IMPLEMENTATIONS[index]
    && stage?.stage_id === GUIDED_STAGE_IDS[index]
    && !stage?.parallel_group
    && Object.keys(options).every((key) => GUIDED_OPTION_KEYS[index].has(key));
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum, fallback) {
  return Math.min(maximum, Math.max(minimum, finiteNumber(value, fallback)));
}

function knownMotionSources(value) {
  const values = typeof value === "string" ? [value] : Array.isArray(value) ? value : [];
  return [...new Set(values
    .map((source) => String(source).trim().toLowerCase())
    .filter((source) => source === "mog2" || source === "onvif"))];
}

export function defaultMotionDecisionSettings() {
  return structuredClone(DEFAULT_SETTINGS);
}

export function motionValidatorSettings(
  current,
  { mode, adaptiveEnabled = true, mog2Enabled = false, agreement = "all" },
) {
  const includePrimary = ["adaptive", "camera_rescue"].includes(mode)
    ? true
    : Boolean(adaptiveEnabled);
  const sources = mog2Enabled ? ["mog2"] : [];
  let policy = "bypass";
  if (includePrimary && !mog2Enabled) policy = "audit";
  else if (mog2Enabled && !includePrimary) policy = "all";
  else if (includePrimary && mog2Enabled) {
    // In visual-triggered mode MOG2 may corroborate adaptive motion, but it
    // must never become an independent trigger by rescuing a rejected primary.
    policy = mode === "adaptive" ? "all" : agreement === "any" ? "any" : "all";
  }
  return {
    ...current,
    policy,
    sources,
    includePrimary,
    failOpen: true,
  };
}

export function readMotionDecisionFusion(fusion) {
  if (fusion == null || (Array.isArray(fusion) && fusion.length === 0)) {
    return { custom: false, usesDefaults: true, settings: defaultMotionDecisionSettings() };
  }
  if (
    !Array.isArray(fusion)
    || fusion.length !== GUIDED_IMPLEMENTATIONS.length
    || fusion.some((stage, index) => !isGuidedStage(stage, index))
    || (fusion[0]?.options?.minimum_sources ?? 1) !== 1
    || (fusion[0]?.options?.require_warmed ?? true) !== true
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
      policy: ["audit", "bypass", "any", "all", "weighted"].includes(
        String(fusionOptions.policy || "").trim().toLowerCase(),
      )
        ? String(fusionOptions.policy).trim().toLowerCase()
        : defaults.policy,
      sources: knownMotionSources(fusionOptions.sources),
      includePrimary: fusionOptions.include_primary ?? defaults.includePrimary,
      failOpen: fusionOptions.fail_open ?? defaults.failOpen,
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
  const sources = knownMotionSources(normalized.sources);
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
        include_primary: normalized.includePrimary !== false,
        fail_open: normalized.failOpen !== false,
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
