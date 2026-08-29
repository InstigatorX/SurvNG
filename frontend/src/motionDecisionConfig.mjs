const DEFAULT_SETTINGS = Object.freeze({
  policy: "audit",
  sources: [],
  includePrimary: true,
  failOpen: true,
  sourceThresholds: { onvif: 0.5 },
  sourceWeights: { primary: 1, onvif: 1 },
  weightedThreshold: 0.5,
});

export const MOTION_MODE_OPTIONS = Object.freeze([
  Object.freeze({
    value: "camera",
    label: "Camera-triggered",
    status: "Camera ONVIF triggers",
    description: "Only camera ONVIF notices and manual tests can start object detection. Fastest when notices are trusted. Optional visual validators can confirm ordinary camera motion before detection runs.",
  }),
  Object.freeze({
    value: "camera_rescue",
    label: "Camera + EMA backup (Recommended)",
    status: "Camera primary · EMA backup",
    description: "Camera ONVIF notices remain primary and skip EMA filtering. Continuous EMA can still rescue a silent camera; an eligible object must overlap that motion or move across detector samples. Pays extra CPU for recall, not for fewer false positives.",
  }),
  Object.freeze({
    value: "adaptive",
    label: "EMA-triggered",
    status: "SurvNG EMA triggers",
    description: "Enhanced Motion Analysis (EMA) starts object detection. Ordinary ONVIF notices are recorded as diagnostics but cannot trigger detection.",
  }),
]);

export const MOTION_BEHAVIOR_OPTIONS = Object.freeze([
  Object.freeze({
    value: "camera",
    label: "Camera only",
    status: "Camera notices trigger detection",
    description: "Every ordinary ONVIF motion notice proceeds directly to object detection. EMA does not validate or independently trigger detection. Use this when camera notices are trusted and you want minimum extra CPU.",
    mode: "camera",
    adaptiveEnabled: false,
  }),
  Object.freeze({
    value: "camera_validation",
    label: "Camera + EMA validation",
    status: "Camera trigger · EMA validation",
    description: "ONVIF is required to start the event. EMA validates ordinary camera motion before object detection runs, which can cut noisy false notices, but cannot trigger detection by itself. Pays continuous analysis CPU plus a short wait.",
    mode: "camera",
    adaptiveEnabled: true,
  }),
  Object.freeze({
    value: "camera_rescue",
    label: "Camera + EMA backup",
    status: "Camera primary · EMA backup",
    description: "ONVIF starts object detection immediately. Strong, persistent EMA motion may independently rescue a missing camera notice using additional safeguards. Use this so silent cameras are not missed; it does not filter ONVIF false positives.",
    mode: "camera_rescue",
    adaptiveEnabled: true,
  }),
  Object.freeze({
    value: "adaptive",
    label: "EMA only",
    status: "EMA triggers detection",
    description: "EMA is the automatic trigger. ONVIF notices remain diagnostic and cannot start object detection.",
    mode: "adaptive",
    adaptiveEnabled: true,
  }),
]);

export function motionBehaviorOption(value) {
  return MOTION_BEHAVIOR_OPTIONS.find((option) => option.value === value)
    || MOTION_BEHAVIOR_OPTIONS[2];
}

export function motionBehaviorValue(mode, settings) {
  if (mode === "camera_rescue") return "camera_rescue";
  if (mode === "adaptive") return "adaptive";
  if (mode === "camera") {
    return settings?.policy === "bypass" || settings?.includePrimary === false
      ? "camera"
      : "camera_validation";
  }
  return String(mode || "camera_rescue");
}

export function motionBehaviorSettings(current, behavior) {
  const option = motionBehaviorOption(behavior);
  return {
    mode: option.mode,
    settings: motionValidatorSettings(current, {
      mode: option.mode,
      adaptiveEnabled: option.adaptiveEnabled,
    }),
  };
}

const LEGACY_MODE_INFO = Object.freeze({
  audit: Object.freeze({
    value: "audit",
    label: "Legacy decision preview",
    status: "Legacy camera triggers · preview only",
    description: "Camera notices always run object detection; EMA decisions are recorded but never enforced. Select a current mode to migrate this configuration.",
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
    description: "Both ordinary ONVIF and EMA may trigger detection. Select a current mode to remove this ambiguous hybrid behavior.",
  }),
});

export function motionModeInfo(mode) {
  if (LEGACY_MODE_INFO[mode]) return LEGACY_MODE_INFO[mode];
  return MOTION_MODE_OPTIONS.find((option) => option.value === mode)
    || MOTION_MODE_OPTIONS.find((option) => option.value === "camera_rescue");
}

const FUSION_OPTION_KEYS = new Set([
    "sources",
    "policy",
    "source_thresholds",
    "source_weights",
    "weighted_threshold",
    "minimum_sources",
    "require_warmed",
    "include_primary",
    "fail_open",
  ]);

function isGuidedFusionStage(stage) {
  const options = stage?.options || {};
  return stage?.implementation === "buffered_evidence_fusion"
    && stage?.stage_id === "evidence_fusion"
    && !stage?.parallel_group
    && Object.keys(options).every((key) => FUSION_OPTION_KEYS.has(key));
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
    .filter((source) => source === "onvif"))];
}

export function defaultMotionDecisionSettings() {
  return structuredClone(DEFAULT_SETTINGS);
}

export function motionValidatorSettings(
  current,
  { mode, adaptiveEnabled = true },
) {
  const includePrimary = ["adaptive", "camera_rescue"].includes(mode)
    ? true
    : Boolean(adaptiveEnabled);
  const policy = includePrimary ? "audit" : "bypass";
  return {
    ...current,
    policy,
    sources: [],
    includePrimary,
    failOpen: true,
  };
}

export function readMotionDecisionFusion(fusion) {
  if (fusion == null || (Array.isArray(fusion) && fusion.length === 0)) {
    return { custom: false, usesDefaults: true, settings: defaultMotionDecisionSettings() };
  }
  const migrated = Array.isArray(fusion)
    ? fusion.filter((stage) => !["score_event_state", "score_trigger"].includes(stage?.implementation))
    : fusion;
  if (
    !Array.isArray(migrated)
    || migrated.length !== 1
    || !isGuidedFusionStage(migrated[0])
    || (migrated[0]?.options?.minimum_sources ?? 1) !== 1
    || (migrated[0]?.options?.require_warmed ?? true) !== true
  ) {
    return { custom: true, usesDefaults: false, settings: defaultMotionDecisionSettings() };
  }

  const fusionOptions = migrated[0]?.options || {};
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
        onvif: clamp(fusionOptions.source_thresholds?.onvif, 0, 1, defaults.sourceThresholds.onvif),
      },
      sourceWeights: {
        primary: clamp(fusionOptions.source_weights?.primary, 0, 10, defaults.sourceWeights.primary),
        onvif: clamp(fusionOptions.source_weights?.onvif, 0, 10, defaults.sourceWeights.onvif),
      },
      weightedThreshold: clamp(
        fusionOptions.weighted_threshold,
        0,
        1,
        defaults.weightedThreshold,
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
          onvif: clamp(normalized.sourceThresholds?.onvif, 0, 1, 0.5),
        },
        source_weights: {
          primary: clamp(normalized.sourceWeights?.primary, 0, 10, 1),
          onvif: clamp(normalized.sourceWeights?.onvif, 0, 10, 1),
        },
        weighted_threshold: clamp(normalized.weightedThreshold, 0, 1, 0.5),
        minimum_sources: 1,
        require_warmed: true,
        include_primary: normalized.includePrimary !== false,
        fail_open: normalized.failOpen !== false,
      },
    },
  ];
}
