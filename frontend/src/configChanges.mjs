// Describe unsaved Admin changes as readable "old → new" rows, and undo one
// change at a time. Used by the "Review before saving" panel and the camera
// list's unsaved markers. Values are redacted before display.

const SECRET_KEY = /(password|secret|token|api_?key|private_?key|credential)/i;
const URL_CREDENTIALS = /(\b[a-z][a-z0-9+.-]*:\/\/)([^/@\s]+)@/gi;
const SECRET_QUERY = /([?&;](?:password|passwd|pwd|pass|token|api_?key|key|secret|auth|signature)[^=&;#]*=)[^&;#]*/gi;
const ACRONYMS = new Map([
  ["url", "URL"], ["rtsp", "RTSP"], ["onvif", "ONVIF"], ["fps", "FPS"], ["mqtt", "MQTT"],
  ["api", "API"], ["id", "ID"], ["ptz", "PTZ"], ["hls", "HLS"], ["gpu", "GPU"], ["cpu", "CPU"],
  ["ema", "EMA"], ["tls", "TLS"], ["http", "HTTP"], ["https", "HTTPS"], ["homekit", "HomeKit"],
]);

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function equal(left, right) {
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

export function humanizeKey(key) {
  const words = String(key).replace(/([a-z])([A-Z])/g, "$1 $2").split(/[_\s-]+/).filter(Boolean);
  return words.map((word, index) => {
    const lower = word.toLowerCase();
    if (ACRONYMS.has(lower)) return ACRONYMS.get(lower);
    return index === 0 ? lower.charAt(0).toUpperCase() + lower.slice(1) : lower;
  }).join(" ");
}

export function changeLabel(path) {
  return path.map(humanizeKey).join(" › ");
}

/** Display form of a value; secrets and URL credentials never leave as text. */
export function formatChangeValue(value, path = []) {
  if (value === undefined) return "not set";
  if (path.some((segment) => SECRET_KEY.test(String(segment)))) return value ? "hidden" : "empty";
  if (value === null || value === "") return "empty";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  if (typeof value === "string") return value.replace(URL_CREDENTIALS, "$1***@").replace(SECRET_QUERY, "$1***");
  if (Array.isArray(value)) {
    if (!value.length) return "none";
    if (value.every((item) => ["string", "number"].includes(typeof item))) return value.join(", ");
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }
  return "changed";
}

function zoneKey(zone, index) {
  return String(zone?.id || zone?.name || `#${index}`);
}

function zoneChanges(before = [], after = []) {
  const beforeByKey = new Map(before.map((zone, index) => [zoneKey(zone, index), zone]));
  const afterByKey = new Map(after.map((zone, index) => [zoneKey(zone, index), zone]));
  const changes = [];
  for (const [key, zone] of afterByKey) {
    const name = zone?.name || key;
    if (!beforeByKey.has(key)) changes.push({ path: ["zones"], zone: key, label: "Zones", kind: "added", text: `+ ${name}` });
    else if (!equal(beforeByKey.get(key), zone)) changes.push({ path: ["zones"], zone: key, label: "Zones", kind: "changed", text: `${name} edited` });
  }
  for (const [key, zone] of beforeByKey) {
    if (!afterByKey.has(key)) changes.push({ path: ["zones"], zone: key, label: "Zones", kind: "removed", text: `− ${zone?.name || key}` });
  }
  if (!changes.length && !equal(before, after)) {
    changes.push({ path: ["zones"], zone: null, label: "Zones", kind: "changed", text: "Order changed" });
  }
  return changes;
}

/** Leaf-level differences between two values, as path rows. */
export function diffValues(before, after, path = []) {
  if (equal(before, after)) return [];
  if (path.length === 1 && path[0] === "zones" && (Array.isArray(before) || Array.isArray(after))) {
    return zoneChanges(Array.isArray(before) ? before : [], Array.isArray(after) ? after : []);
  }
  if (isPlainObject(before) && isPlainObject(after)) {
    const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])];
    return keys.flatMap((key) => diffValues(before[key], after[key], [...path, key]));
  }
  return [{
    path,
    label: changeLabel(path),
    kind: before === undefined ? "added" : after === undefined ? "removed" : "value",
    before: formatChangeValue(before, path),
    after: formatChangeValue(after, path),
  }];
}

/**
 * Unsaved changes grouped for review.
 * cameras: one group per camera with at least one change (new cameras flagged);
 * removed: cameras deleted from the draft; system: non-camera settings;
 * order: true when only the camera order changed for existing cameras.
 */
export function describeConfigChanges(baseline, current) {
  const baselineCameras = Array.isArray(baseline?.cameras) ? baseline.cameras : [];
  const currentCameras = Array.isArray(current?.cameras) ? current.cameras : [];
  const baselineById = new Map(baselineCameras.map((camera) => [camera.id, camera]));
  const currentIds = new Set(currentCameras.map((camera) => camera.id));
  const cameras = [];
  for (const camera of currentCameras) {
    const previous = baselineById.get(camera.id);
    const name = camera.name || previous?.name || camera.id;
    if (!previous) {
      cameras.push({ cameraId: camera.id, name, isNew: true, changes: [{ path: [], label: "Camera", kind: "added", before: "not set", after: "New camera" }] });
      continue;
    }
    const changes = diffValues(previous, camera).map((change) => ({ ...change, cameraId: camera.id }));
    if (changes.length) cameras.push({ cameraId: camera.id, name, isNew: false, changes });
  }
  const removed = baselineCameras.filter((camera) => !currentIds.has(camera.id))
    .map((camera) => ({ cameraId: camera.id, name: camera.name || camera.id }));
  const { cameras: _baselineCameras, ...baselineSystem } = baseline || {};
  const { cameras: _currentCameras, ...currentSystem } = current || {};
  const system = diffValues(baselineSystem, currentSystem);
  const sharedOrder = (list) => list.map((camera) => camera.id).filter((id) => baselineById.has(id) && currentIds.has(id));
  const order = !equal(sharedOrder(baselineCameras), sharedOrder(currentCameras));
  const count = cameras.reduce((total, group) => total + group.changes.length, 0) + removed.length + system.length + (order ? 1 : 0);
  return { cameras, removed, system, order, count };
}

function cloneValue(value) {
  return value === undefined ? undefined : structuredClone(value);
}

function setAtPath(target, path, value) {
  if (!path.length) return value;
  const [head, ...rest] = path;
  const base = isPlainObject(target) ? { ...target } : {};
  const next = setAtPath(base[head], rest, value);
  if (next === undefined) delete base[head];
  else base[head] = next;
  return base;
}

function getAtPath(target, path) {
  return path.reduce((value, key) => (value == null ? undefined : value[key]), target);
}

/**
 * Undo one change in the draft config. Camera changes use change.cameraId;
 * zone rows restore (or drop) just that zone; a new camera is removed.
 */
export function revertConfigChange(current, baseline, change) {
  if (!current || !change) return current;
  if (!change.cameraId) {
    return setAtPath(current, change.path, cloneValue(getAtPath(baseline, change.path)));
  }
  const cameras = Array.isArray(current.cameras) ? current.cameras : [];
  const previous = (baseline?.cameras || []).find((camera) => camera.id === change.cameraId);
  if (!previous) return { ...current, cameras: cameras.filter((camera) => camera.id !== change.cameraId) };
  return {
    ...current,
    cameras: cameras.map((camera) => {
      if (camera.id !== change.cameraId) return camera;
      if (change.path[0] === "zones" && change.zone) {
        const previousZones = previous.zones || [];
        const restored = previousZones.find((zone, index) => zoneKey(zone, index) === change.zone);
        const zones = (camera.zones || []).filter((zone, index) => zoneKey(zone, index) !== change.zone);
        if (!restored) return { ...camera, zones };
        const originalIndex = previousZones.indexOf(restored);
        zones.splice(Math.min(originalIndex, zones.length), 0, structuredClone(restored));
        return { ...camera, zones };
      }
      return setAtPath(camera, change.path, cloneValue(getAtPath(previous, change.path)));
    }),
  };
}

function reviewCount(review) {
  return review.cameras.reduce((total, group) => total + group.changes.length, 0)
    + review.removed.length + review.system.length + (review.order ? 1 : 0);
}

/**
 * The part of a review that one Admin save button applies. Camera settings,
 * zones, removals and order save from the Cameras workspace; system settings
 * and per-camera speech switches save with the system settings.
 */
export function scopeConfigReview(review, scope) {
  if (!review) return null;
  let scoped;
  if (scope === "cameras") {
    scoped = { cameras: review.cameras, removed: review.removed, system: [], order: review.order };
  } else if (scope === "general") {
    const cameras = review.cameras
      .filter((group) => !group.isNew)
      .map((group) => ({ ...group, changes: group.changes.filter((change) => change.path[0] === "speech") }))
      .filter((group) => group.changes.length);
    scoped = { cameras, removed: [], system: review.system, order: false };
  } else {
    scoped = { cameras: [], removed: [], system: [], order: false };
  }
  return { ...scoped, count: reviewCount(scoped) };
}
