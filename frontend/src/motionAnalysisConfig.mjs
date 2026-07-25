function normalizedValue(value) {
  if (Array.isArray(value)) return value.map(normalizedValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, normalizedValue(value[key])]),
    );
  }
  return value;
}

function normalizedGraph(stages) {
  return (stages || []).map((stage) => ({
    stage_id: stage.stage_id,
    implementation: stage.implementation,
    options: normalizedValue(stage.options || {}),
  }));
}

export function availableQualificationPresets(catalog) {
  return (catalog?.presets || []).filter(
    (preset) => preset.graph === "qualification" && preset.available !== false,
  );
}

export function readMotionAnalysisPreset(qualification, catalog) {
  const presets = availableQualificationPresets(catalog);
  const recommended = presets.find((preset) => preset.recommended) || presets[0] || null;
  if (qualification == null) {
    return { inherited: true, custom: false, preset: recommended };
  }
  if (Array.isArray(qualification) && qualification.length === 0) {
    return { inherited: false, custom: false, preset: recommended };
  }
  const normalized = JSON.stringify(normalizedGraph(qualification));
  const preset = presets.find(
    (candidate) => JSON.stringify(normalizedGraph(candidate.stages)) === normalized,
  );
  return { inherited: false, custom: !preset, preset: preset || recommended };
}

export function presetQualificationGraph(preset) {
  return structuredClone(preset?.stages || []);
}
